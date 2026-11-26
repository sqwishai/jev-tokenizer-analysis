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
USER_AGENT = "jev-tokenizer-analysis/0.1 (benchmark research script)"

CODE_SOURCES: list[tuple[str, str]] = [
    (
        "C# / .NET (dotnet/runtime HttpClient.cs)",
        "https://raw.githubusercontent.com/dotnet/runtime/main/src/libraries/System.Net.Http/src/System/Net/Http/HttpClient.cs",
    ),
    (
        "TypeScript (microsoft/vscode codeEditorWidget.ts)",
        "https://raw.githubusercontent.com/microsoft/vscode/main/src/vs/editor/browser/widget/codeEditor/codeEditorWidget.ts",
    ),
    (
        "Objective-C (AFNetworking AFHTTPSessionManager.m)",
        "https://raw.githubusercontent.com/AFNetworking/AFNetworking/master/AFNetworking/AFHTTPSessionManager.m",
    ),
    (
        "Objective-C (SDWebImage SDWebImageManager.m)",
        "https://raw.githubusercontent.com/SDWebImage/SDWebImage/master/SDWebImage/Core/SDWebImageManager.m",
    ),
    (
        "React / JS (facebook/react ReactFiberWorkLoop.js)",
        "https://raw.githubusercontent.com/facebook/react/main/packages/react-reconciler/src/ReactFiberWorkLoop.js",
    ),
    (
        "Java / Spring (AbstractAutowireCapableBeanFactory.java)",
        "https://raw.githubusercontent.com/spring-projects/spring-framework/main/spring-beans/src/main/java/org/springframework/beans/factory/support/AbstractAutowireCapableBeanFactory.java",
    ),
    (
        "Rust (BurntSushi/ripgrep searcher.rs)",
        "https://raw.githubusercontent.com/BurntSushi/ripgrep/master/crates/searcher/src/searcher/mod.rs",
    ),
    (
        "Python (python/cpython argparse.py)",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/argparse.py",
    ),
]

# Real Wikipedia articles on "Earth" (or closely matched general encyclopedia articles)
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


def fetch_code(url: str, max_chars: int = 30_000) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return r.text[:max_chars]


def fetch_wiki(lang: str, title: str, max_chars: int = 8_000) -> tuple[str, str]:
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    for attempt in range(4):
        r = requests.get(
            api_url,
            headers={"User-Agent": "JevTokenizerAnalysis/0.1 (https://github.com/sqwishai/jev-tokenizer-analysis)"},
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

    def eval_corpus(label: str, kind: str, source_url: str, text: str) -> dict:
        q = len(qtok.encode(text, add_special_tokens=False).ids)
        o = len(o200k.encode(text, disallowed_special=()))
        j = jev.count(text)
        row: dict = {
            "workload": label,
            "kind": kind,
            "source_url": source_url,
            "chars": len(text),
            "utf8_bytes": len(text.encode("utf-8")),
            "qwen35_tokens": q,
            "o200k_tokens": o,
            "jev_tokens": j,
            "ratio_vs_qwen35": round(j / q, 3),
            "pct_vs_qwen35": round((j / q - 1) * 100, 1),
            "ratio_vs_o200k": round(j / o, 3),
            "pct_vs_o200k": round((j / o - 1) * 100, 1),
        }
        if api_key:
            # Spot-verify a 1,500-char prefix live against OpenRouter jev-1.14.
            sample = text[:1500]
            pred_s = jev.count(sample)
            live_s = query_live(api_key, sample)
            row["live_spotcheck_chars"] = len(sample)
            row["live_spotcheck_pred"] = pred_s
            row["live_spotcheck_actual"] = live_s
            row["live_spotcheck_match"] = live_s == pred_s
        return row

    fetched: list[tuple[str, str, str, str]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        code_futs = [(label, "code", url, pool.submit(fetch_code, url)) for label, url in CODE_SOURCES]
        for label, kind, url, fut in code_futs:
            fetched.append((label, kind, url, fut.result()))
    for lang_name, lang, title in WIKI_SOURCES:
        txt, page_url = fetch_wiki(lang, title)
        fetched.append((f"{lang_name} Wikipedia ({title})", "wikipedia", page_url, txt))
        time.sleep(0.35)

    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda item: eval_corpus(*item), fetched))

    code_rows = sorted([r for r in rows if r["kind"] == "code"], key=lambda r: r["ratio_vs_qwen35"], reverse=True)
    wiki_rows = sorted([r for r in rows if r["kind"] == "wikipedia"], key=lambda r: r["ratio_vs_qwen35"], reverse=True)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "live_spotcheck_enabled": bool(api_key),
        "live_spotcheck_matched": sum(1 for r in rows if r.get("live_spotcheck_match")) if api_key else None,
        "live_spotcheck_total": len(rows) if api_key else None,
        "code_benchmarks": code_rows,
        "wikipedia_benchmarks": wiki_rows,
    }

    out_path = ROOT / "evidence" / "workload_benchmarks.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for section_title, section_rows in [("REAL OPEN-SOURCE CODE (first 30 KB)", code_rows), ("REAL WIKIPEDIA ARTICLES (first 8,000 chars)", wiki_rows)]:
        print(f"\n=== {section_title} ===")
        print(f"{'Workload':54} {'Qwen3.5':>7} {'o200k':>7} {'Jev':>7} | {'vs Qwen':>14} | {'vs o200k':>9} | live")
        print("-" * 112)
        for r in section_rows:
            lv = "OK" if r.get("live_spotcheck_match") else ("-" if not api_key else "DIFF")
            print(
                f"{r['workload']:54} {r['qwen35_tokens']:7d} {r['o200k_tokens']:7d} {r['jev_tokens']:7d} | "
                f"{r['ratio_vs_qwen35']:5.2f}x ({r['pct_vs_qwen35']:+6.1f}%) | {r['pct_vs_o200k']:+8.1f}% | {lv}"
            )

    if api_key:
        mismatches = [r for r in rows if not r["live_spotcheck_match"]]
        assert not mismatches, f"Live spotcheck mismatches: {mismatches}"
        print(f"\nALL {len(rows)}/{len(rows)} REAL-CORPUS LIVE SPOTCHECKS MATCHED JEV 1.14 100%.")


if __name__ == "__main__":
    main()
