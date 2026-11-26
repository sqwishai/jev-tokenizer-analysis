from __future__ import annotations

import json
import os
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jev_billing_tokenizer import JevBillingTokenizer

OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"

LIVE_EXTREMES: list[tuple[str, str, str]] = [
    ("global_max_single_token", "C/Java 96-slash divider ('/' * 96)", "/" * 96),
    ("scalable_max_plain_text", "Spaced 70-slash divider x10 ((' ' + '/' * 70) * 10)", (" " + "/" * 70) * 10),
    ("scalable_16x_both", "Em-dash run ('—' * 160)", "—" * 160),
    ("scalable_16x_both", "Tilde run ('~' * 320)", "~" * 320),
    ("scalable_16x_both", "Underscore run ('_' * 640)", "_" * 640),
    ("scalable_16x_both", "Slash run ('/' * 960)", "/" * 960),
    ("scalable_16x_qwen", "Percent run ('%' * 640)", "%" * 640),
    ("code_block_banner", "97-char C block header ('/' + '*' * 96)", "/" + "*" * 96),
    ("real_code_csharp", "C# / .NET: .DataGridViewColumnHeadersHeightSizeMode", ".DataGridViewColumnHeadersHeightSizeMode"),
    ("real_code_objc", "Objective-C / iOS: ' dequeueReusableCellWithIdentifier'", " dequeueReusableCellWithIdentifier"),
    ("real_code_objc", "Objective-C / iOS: .translatesAutoresizingMaskIntoConstraints", ".translatesAutoresizingMaskIntoConstraints"),
    ("real_code_android", "Android / Java: NavigationItemSelectedListener", "NavigationItemSelectedListener"),
    ("real_code_java", "Java / Android: ' UnsupportedOperationException'", " UnsupportedOperationException"),
    ("real_code_js", "JS / Jest: toHaveBeenCalledWith", "toHaveBeenCalledWith"),
    ("real_lang_thai", "Thai ('more info'): ข้อมูลเพิ่มเติม", "ข้อมูลเพิ่มเติม"),
    ("real_lang_thai", "Thai ('privacy'): ความเป็นส่วนตัว", "ความเป็นส่วนตัว"),
    ("real_lang_chinese", "Chinese footer: 转载请附上原文出处链接和本声明", "转载请附上原文出处链接和本声明"),
    ("real_lang_russian", "Russian ('financing'): ' финансирования'", " финансирования"),
    ("real_lang_russian", "Russian ('economic'): ' экономического'", " экономического"),
    ("real_lang_arabic", "Arabic ('strategic'): ' الاستراتيجية'", " الاستراتيجية"),
    ("underbill_floor", "Repeated 'ab' * 100 (0.50x underbilling floor)", "ab" * 100),
    ("underbill_floor", "15-hyphen newline ('-' * 15 + '\\n', 0.50x)", "-" * 15 + "\n"),
]


def query_live(api_key: str, text: str) -> int:
    r = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "typesafe/jev-1.14",
            "state": text,
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["usage"]["input_tokens"] - 267


def main() -> None:
    jev = JevBillingTokenizer()
    qtok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3.5-9B", "tokenizer.json"))
    o200k = jev.o200k
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    # 1. Exhaustive scan of all 247,843 reachable Qwen 3.5 tokens (single-token overbilling max)
    single_token_max = []
    for bs in jev.qwen_reachable:
        if len(bs) < 10:
            continue
        try:
            s = bs.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if unicodedata.normalize("NFC", s) != s:
            continue
        if jev.pretokenizer.findall(s) != [s]:
            continue
        if len(qtok.encode(s, add_special_tokens=False).ids) != 1:
            continue
        jc = jev.count(s)
        if jc >= 12:
            oc = len(o200k.encode(s, disallowed_special=()))
            single_token_max.append({"jev_tokens": jc, "qwen35_tokens": 1, "o200k_tokens": oc, "utf8_bytes": len(bs), "text": s})
    single_token_max.sort(key=lambda r: (r["jev_tokens"], r["utf8_bytes"]), reverse=True)

    # 2. Exhaustive scan of all 53,851 FORMED tokens (maximum underbilling Qwen / Jev)
    max_underbill = 1.0
    underbill_examples = []
    for bs in jev.pass2_formed_ranks:
        try:
            s = bs.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if unicodedata.normalize("NFC", s) != s:
            continue
        jc = jev.count(s)
        qc = len(qtok.encode(s, add_special_tokens=False).ids)
        if qc > jc:
            ratio = qc / jc
            if ratio > max_underbill:
                max_underbill = ratio
            if ratio >= 2.0 and len(underbill_examples) < 10:
                underbill_examples.append({"qwen_over_jev": ratio, "jev_tokens": jc, "qwen35_tokens": qc, "text": s})

    # 3. Live-verify curated extremal strings on jev-1.14
    live_counts: dict[str, int] = {}
    if api_key:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for label, cnt in pool.map(lambda it: (it[1], query_live(api_key, it[2])), LIVE_EXTREMES):
                live_counts[label] = cnt

    extremes_rows = []
    for cat, label, txt in LIVE_EXTREMES:
        qc = len(qtok.encode(txt, add_special_tokens=False).ids)
        oc = len(o200k.encode(txt, disallowed_special=()))
        pred = jev.count(txt)
        row: dict = {
            "category": cat,
            "label": label,
            "qwen35_tokens": qc,
            "o200k_tokens": oc,
            "jev_predicted_tokens": pred,
            "jev_over_qwen35": round(pred / qc, 3),
            "jev_over_o200k": round(pred / oc, 3),
        }
        if api_key:
            row["jev_live_tokens"] = live_counts[label]
            row["live_match"] = live_counts[label] == pred
        extremes_rows.append(row)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scanned_qwen35_reachable_tokens": len(jev.qwen_reachable),
        "scanned_pass2_formed_tokens": len(jev.pass2_formed_ranks),
        "global_single_token_max_overbill_ratio": single_token_max[0]["jev_tokens"],
        "global_single_token_max_underbill_qwen_over_jev": max_underbill,
        "live_verified": bool(api_key),
        "live_matched": sum(1 for r in extremes_rows if r.get("live_match")) if api_key else None,
        "live_total": len(extremes_rows) if api_key else None,
        "top_single_qwen_tokens": single_token_max[:25],
        "underbill_floor_examples": underbill_examples,
        "curated_extremes": extremes_rows,
    }

    out_path = ROOT / "evidence" / "extremal_multipliers.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"Global single-token max overbilling: {report['global_single_token_max_overbill_ratio']}x | "
        f"Global single-token max underbilling: 1/{max_underbill:.1f}x ({1/max_underbill:.2f}x)"
    )
    for r in extremes_rows:
        lv = str(r["jev_live_tokens"]) if api_key else "-"
        print(
            f"  {r['jev_over_qwen35']:6.2f}x vs Qwen ({r['jev_over_o200k']:5.2f}x vs o200k) "
            f"| pred={r['jev_predicted_tokens']:4d} live={lv:>4} Qwen={r['qwen35_tokens']:3d} | {r['label']}"
        )

    if api_key:
        mismatches = [r for r in extremes_rows if not r["live_match"]]
        assert not mismatches, f"Mismatches: {mismatches}"
        print(f"\nALL {len(extremes_rows)}/{len(extremes_rows)} EXTREMAL MULTIPLIERS MATCHED LIVE JEV 1.14 100%.")


if __name__ == "__main__":
    main()
