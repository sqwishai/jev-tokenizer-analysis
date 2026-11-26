from __future__ import annotations

import json
import os
import sys
import time
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
USER_AGENT = "JevTokenizerAnalysis/0.1 (https://github.com/sqwishai/jev-tokenizer-analysis)"

CODE_SOURCES: list[tuple[str, str]] = [
    (
        "Objective-C / iOS (Instagram/IGListKit IGListAdapter.m)",
        "https://raw.githubusercontent.com/Instagram/IGListKit/main/Source/IGListKit/IGListAdapter.m",
    ),
    (
        "Java / Spring (RequestMappingHandlerAdapter.java)",
        "https://raw.githubusercontent.com/spring-projects/spring-framework/main/spring-webmvc/src/main/java/org/springframework/web/servlet/mvc/method/annotation/RequestMappingHandlerAdapter.java",
    ),
    (
        "Android Material (TextInputLayout.java)",
        "https://raw.githubusercontent.com/material-components/material-components-android/master/lib/java/com/google/android/material/textfield/TextInputLayout.java",
    ),
    (
        "C# / .NET (dotnet/runtime HttpClient.cs)",
        "https://raw.githubusercontent.com/dotnet/runtime/main/src/libraries/System.Net.Http/src/System/Net/Http/HttpClient.cs",
    ),
    (
        "Android Material (NavigationBarView.java)",
        "https://raw.githubusercontent.com/material-components/material-components-android/master/lib/java/com/google/android/material/navigation/NavigationBarView.java",
    ),
    (
        "Objective-C / iOS (AFNetworking AFHTTPSessionManager.m)",
        "https://raw.githubusercontent.com/AFNetworking/AFNetworking/master/AFNetworking/AFHTTPSessionManager.m",
    ),
    (
        "C# / .NET WinForms (DataGridView.cs)",
        "https://raw.githubusercontent.com/dotnet/winforms/main/src/System.Windows.Forms/System/Windows/Forms/Controls/DataGridView/DataGridView.cs",
    ),
    (
        "Android Jetpack Compose / Kotlin (AndroidComposeView.android.kt)",
        "https://raw.githubusercontent.com/androidx/androidx/androidx-main/compose/ui/ui/src/androidMain/kotlin/androidx/compose/ui/platform/AndroidComposeView.android.kt",
    ),
    (
        "TypeScript (microsoft/vscode codeEditorWidget.ts)",
        "https://raw.githubusercontent.com/microsoft/vscode/main/src/vs/editor/browser/widget/codeEditor/codeEditorWidget.ts",
    ),
    (
        "AndroidX (RecyclerView.java)",
        "https://raw.githubusercontent.com/androidx/androidx/androidx-main/recyclerview/recyclerview/src/main/java/androidx/recyclerview/widget/RecyclerView.java",
    ),
    (
        "Swift / iOS (Alamofire Session.swift)",
        "https://raw.githubusercontent.com/Alamofire/Alamofire/master/Source/Core/Session.swift",
    ),
    (
        "React / JS (facebook/react ReactFiberWorkLoop.js)",
        "https://raw.githubusercontent.com/facebook/react/main/packages/react-reconciler/src/ReactFiberWorkLoop.js",
    ),
]

WIKI_SOURCES: list[tuple[str, str, str]] = [
    ("Thai", "th", "โลก"),
    ("Russian", "ru", "Земля"),
    ("Ukrainian", "uk", "Земля"),
    ("Arabic", "ar", "الأرض"),
    ("Greek", "el", "Γη"),
    ("Chinese (Simplified)", "zh", "地球"),
    ("Japanese", "ja", "地球"),
    ("Hebrew", "he", "כדור הארץ"),
    ("Hindi (Devanagari)", "hi", "पृथ्वी"),
    ("Polish", "pl", "Ziemia"),
    ("German", "de", "Erde"),
    ("Turkish", "tr", "Dünya"),
    ("Korean (Hangul)", "ko", "지구"),
    ("Czech", "cs", "Země"),
    ("Vietnamese", "vi", "Trái Đất"),
    ("English", "en", "Earth"),
]


def strip_c_comments(code: str) -> str:
    out: list[str] = []
    i = 0
    n = len(code)
    in_str: str | None = None
    while i < n:
        ch = code[i]
        if in_str is not None:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(code[i + 1])
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ('"', "'", "`"):
            in_str = ch
            out.append(ch)
            i += 1
        elif ch == "/" and i + 1 < n and code[i + 1] == "/":
            i += 2
            while i < n and code[i] != "\n":
                i += 1
        elif ch == "/" and i + 1 < n and code[i + 1] == "*":
            i += 2
            while i + 1 < n and not (code[i] == "*" and code[i + 1] == "/"):
                i += 1
            i += 2
        else:
            out.append(ch)
            i += 1
    return "\n".join(ln for ln in "".join(out).splitlines() if ln.strip())


def fetch_code(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.text


def fetch_wiki(lang: str, title: str, max_chars: int = 8_000) -> tuple[str, str]:
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    for attempt in range(4):
        r = requests.get(
            api_url,
            headers={"User-Agent": USER_AGENT},
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": "1",
                "redirects": "1",
                "titles": title,
                "format": "json",
            },
            timeout=30,
        )
        if r.status_code == 429:
            time.sleep(2.0 * (attempt + 1))
            continue
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        page = next(iter(pages.values()))
        text = page.get("extract", "").strip()
        if len(text) < 500:
            raise RuntimeError(f"Short Wikipedia extract for {lang}:{title} ({len(text)} chars)")
        return text[:max_chars], f"https://{lang}.wikipedia.org/wiki/{title}"
    r.raise_for_status()
    raise RuntimeError(f"Failed to fetch Wikipedia {lang}:{title}")


def query_live(api_key: str, text: str) -> int:
    r = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "typesafe/jev-1.14",
            "state": text,
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["usage"]["input_tokens"] - 267


def main() -> None:
    jev = JevBillingTokenizer()
    qtok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3.5-9B", "tokenizer.json"))
    o200k = jev.o200k
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    def count_all(text: str) -> tuple[int, int, int]:
        return (
            len(qtok.encode(text, add_special_tokens=False).ids),
            len(o200k.encode(text, disallowed_special=())),
            jev.count(text),
        )

    code_fetched: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [(label, url, pool.submit(fetch_code, url)) for label, url in CODE_SOURCES]
        for label, url, fut in futs:
            code_fetched.append((label, url, fut.result()))

    wiki_fetched: list[tuple[str, str, str]] = []
    for lang_name, lang, title in WIKI_SOURCES:
        txt, page_url = fetch_wiki(lang, title)
        wiki_fetched.append((f"{lang_name} Wikipedia ({title})", page_url, txt))
        time.sleep(0.35)

    def eval_code(item: tuple[str, str, str]) -> dict:
        label, url, full_raw = item
        raw_30k = full_raw[:30_000]
        code_30k = strip_c_comments(full_raw)[:30_000]

        q_raw, o_raw, j_raw = count_all(raw_30k)
        q_code, o_code, j_code = count_all(code_30k)
        row: dict = {
            "workload": label,
            "source_url": url,
            "raw_30kb": {
                "chars": len(raw_30k),
                "qwen35_tokens": q_raw,
                "o200k_tokens": o_raw,
                "jev_tokens": j_raw,
                "ratio_vs_qwen35": round(j_raw / q_raw, 3),
                "pct_vs_qwen35": round((j_raw / q_raw - 1) * 100, 1),
                "pct_vs_o200k": round((j_raw / o_raw - 1) * 100, 1),
            },
            "code_only_no_comments_30kb": {
                "chars": len(code_30k),
                "qwen35_tokens": q_code,
                "o200k_tokens": o_code,
                "jev_tokens": j_code,
                "ratio_vs_qwen35": round(j_code / q_code, 3),
                "pct_vs_qwen35": round((j_code / q_code - 1) * 100, 1),
                "ratio_vs_o200k": round(j_code / o_code, 3),
                "pct_vs_o200k": round((j_code / o_code - 1) * 100, 1),
            },
        }
        if api_key:
            sample = code_30k[:1500]
            pred_s = jev.count(sample)
            live_s = query_live(api_key, sample)
            row["live_spotcheck_code_only"] = {
                "chars": len(sample),
                "pred": pred_s,
                "actual": live_s,
                "match": live_s == pred_s,
            }
        return row

    def eval_wiki(item: tuple[str, str, str]) -> dict:
        label, url, txt = item
        q, o, j = count_all(txt)
        row: dict = {
            "workload": label,
            "source_url": url,
            "chars": len(txt),
            "utf8_bytes": len(txt.encode("utf-8")),
            "qwen35_tokens": q,
            "o200k_tokens": o,
            "jev_tokens": j,
            "ratio_vs_qwen35": round(j / q, 3),
            "pct_vs_qwen35": round((j / q - 1) * 100, 1),
            "ratio_vs_o200k": round(j / o, 3),
            "pct_vs_o200k": round((j / o - 1) * 100, 1),
        }
        if api_key:
            sample = txt[:1500]
            pred_s = jev.count(sample)
            live_s = query_live(api_key, sample)
            row["live_spotcheck"] = {
                "chars": len(sample),
                "pred": pred_s,
                "actual": live_s,
                "match": live_s == pred_s,
            }
        return row

    with ThreadPoolExecutor(max_workers=6) as pool:
        code_rows = list(pool.map(eval_code, code_fetched))
        wiki_rows = list(pool.map(eval_wiki, wiki_fetched))

    code_rows.sort(key=lambda r: r["code_only_no_comments_30kb"]["ratio_vs_qwen35"], reverse=True)
    wiki_rows.sort(key=lambda r: r["ratio_vs_qwen35"], reverse=True)

    all_matches = (
        [r["live_spotcheck_code_only"]["match"] for r in code_rows]
        + [r["live_spotcheck"]["match"] for r in wiki_rows]
        if api_key
        else []
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "live_spotcheck_enabled": bool(api_key),
        "live_spotcheck_matched": sum(1 for m in all_matches if m) if api_key else None,
        "live_spotcheck_total": len(all_matches) if api_key else None,
        "code_benchmarks": code_rows,
        "wikipedia_benchmarks": wiki_rows,
    }

    out_path = ROOT / "evidence" / "workload_benchmarks.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== REAL OPEN-SOURCE CODE (30 KB: RAW vs. CODE-ONLY NO COMMENTS) ===")
    print(f"{'Repository / File':64} | {'Raw':>6} | {'Code-only Qwen/o200k/Jev':>24} | {'Code vs Qwen':>15} | {'vs o200k':>8} | live")
    print("-" * 136)
    for r in code_rows:
        raw = r["raw_30kb"]
        co = r["code_only_no_comments_30kb"]
        lv = "OK" if r.get("live_spotcheck_code_only", {}).get("match") else ("-" if not api_key else "DIFF")
        print(
            f"{r['workload']:64} | {raw['ratio_vs_qwen35']:5.2f}x | "
            f"{co['qwen35_tokens']:7d} / {co['o200k_tokens']:6d} / {co['jev_tokens']:6d} | "
            f"{co['ratio_vs_qwen35']:5.2f}x ({co['pct_vs_qwen35']:+6.1f}%) | {co['pct_vs_o200k']:+7.1f}% | {lv}"
        )

    print("\n=== REAL WIKIPEDIA ARTICLES (first 8,000 chars) ===")
    print(f"{'Wikipedia article':54} | {'Qwen3.5':>7} {'o200k':>7} {'Jev':>7} | {'vs Qwen':>15} | {'vs o200k':>8} | live")
    print("-" * 116)
    for r in wiki_rows:
        lv = "OK" if r.get("live_spotcheck", {}).get("match") else ("-" if not api_key else "DIFF")
        print(
            f"{r['workload']:54} | {r['qwen35_tokens']:7d} {r['o200k_tokens']:7d} {r['jev_tokens']:7d} | "
            f"{r['ratio_vs_qwen35']:5.2f}x ({r['pct_vs_qwen35']:+6.1f}%) | {r['pct_vs_o200k']:+7.1f}% | {lv}"
        )

    if api_key:
        assert all(all_matches), "Live spotcheck mismatch!"
        print(f"\nALL {len(all_matches)}/{len(all_matches)} REAL-CORPUS LIVE SPOTCHECKS MATCHED JEV 1.14 100%.")


if __name__ == "__main__":
    main()
