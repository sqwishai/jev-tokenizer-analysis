from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jev_billing_tokenizer import JevBillingTokenizer

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY in the environment to run live probes.")
    return key


def query_jev(
    api_key: str,
    *,
    state: str,
    instructions: str = "Is this empty?",
    question_id: str = "q",
    model: str = "~typesafe/jev-latest",
) -> dict:
    resp = requests.post(
        OPENROUTER_DECISIONS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "state": state,
            "questions": {question_id: {"type": "noul", "instructions": instructions}},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


STATE_PROBES: list[tuple[str, str, str]] = [
    ("baseline", "hello", "In o200k & Qwen; all substrings of 'hell' in Qwen"),
    ("baseline", "helloworld", "Misses Pass 1; 'orld' blocked by missing 'orl'"),
    ("baseline", "JavaScript", "Qwen keeps 1 chunk; Pass 2 splits J|ava|S|crip|t"),
    ("baseline", "Příliš žlutý člun minul třetí bóji.", "Czech diacritics: broken UTF-8 prefixes blocked"),
    ("pretokenizer", "123456", "Qwen \\p{N} splits single digits (6) vs o200k \\p{N}{1,3} (2)"),
    ("pretokenizer", "don't won't I've we're", "Qwen contraction splitting (8) vs o200k (4)"),
    ("normalization", "café", "NFC precomposed"),
    ("normalization", "cafe\u0301", "NFD decomposed normalizes to NFC (2, not o200k's 3)"),
    ("normalization", "\ufb01le", "NFC (not NFKC): fi-ligature does not fold to 'fi'"),
    ("special_tokens", "a <|endoftext|> b", "Special tokens treated as ordinary text"),
    ("special_tokens", "a <|im_start|> b", "Qwen chat tokens treated as ordinary text"),
    ("pass1_lookup", "Java", "Standalone in o200k & Qwen -> 1 despite missing 'Jav'"),
    ("pass1_lookup", "Script", "Standalone in o200k & Qwen -> 1 despite missing 'Scrip'"),
    ("pass1_lookup", "inqu", "Standalone in o200k & Qwen -> 1 despite missing 'nq'"),
    ("pass1_lookup", "errors", "Standalone in o200k & Qwen -> 1 despite missing 'rro'"),
    ("pass1_lookup", "icients", "Standalone in o200k & Qwen -> 1 despite missing 'cien'"),
    ("pass1_lookup", "ierenden", "Standalone in o200k & Qwen -> 1 despite missing 'erend'"),
    ("pass1_lookup", "pload", "Qwen 3.5 extended vocab (ID 192,153) passes Pass 1"),
    ("pass1_lookup", "aaaaaaaa", "8x 'a' passes Pass 1 -> 1"),
    ("pass1_lookup", "aaaaaaaaa", "9x 'a' misses Pass 1; 8x 'a' blocked in Pass 2 -> 3"),
    ("pass1_reject_non_qwen", "oworld", "In o200k, NOT in Qwen -> fails Pass 1 -> 3 (ow|or|ld)"),
    ("pass1_reject_mergeless", "yarakat", "In o200k & q_vocab, NOT in q_merges -> fails Pass 1 -> 2 (Qwen=3)"),
    ("pass1_reject_mergeless", " terdapat", "In o200k & q_vocab, NOT in q_merges -> fails Pass 1 -> 2 (Qwen=3)"),
    ("pass1_reject_mergeless", " dikkat", "In o200k & q_vocab, NOT in q_merges -> fails Pass 1 -> 2 (Qwen=3)"),
    ("pass1_reject_mergeless", " факт", "In o200k & q_vocab, NOT in q_merges -> fails Pass 1 -> 4 (Qwen=3)"),
    ("pass1_reject_mergeless", "матри", "In o200k & q_vocab, NOT in q_merges -> fails Pass 1 -> 5 (Qwen=3)"),
    ("pass1_reject_mergeless", "加拿大", "In o200k & q_vocab (ID 108,918 < 151k), NOT in q_merges -> 3"),
    ("pass2_strict_prefix_missing", "Btnxyz", "3-byte 'Btn': missing [0:2]='Bt' blocks merge"),
    ("pass2_strict_prefix_missing", "createxyz", "Missing prefix 'crea' blocks 'create'"),
    ("pass2_strict_prefix_missing", "thingxyz", "Missing prefix 'thi' blocks 'thing'"),
    ("pass2_strict_prefix_missing", "Eventxyz", "Missing prefix 'Eve' blocks 'Event'"),
    ("pass2_cand_minus_1_missing", "Javaxyz", "Missing cand[:-1]='Jav' blocks 'Java'"),
    ("pass2_suffix_of_m1_missing", "inquxyz", "Sole missing substring [1:3]='nq' blocks 'inqu'"),
    ("pass2_suffix_of_m1_missing", "errorxyz", "Sole missing substring [1:4]='rro' blocks 'error'"),
    ("pass2_suffix_of_m1_missing", "orderxyz", "Sole missing substring [1:4]='rde' blocks 'order'"),
    ("pass2_suffix_of_m1_missing", " minulxyz", "Sole missing substring [1:5]='minu' blocks ' minul'"),
    ("pass2_strict_interior_missing", "errorsxyz", "All prefixes & suffixes of 'error' in Q; sole missing [1:4]='rro'"),
    ("pass2_strict_interior_missing", "weightsxyz", "All prefixes & suffixes of 'weight' in Q; missing 'eig','eigh'"),
    ("pass2_strict_interior_missing", "indicesxyz", "All prefixes & suffixes of 'indice' in Q; missing 'ndi','ndic'"),
    ("pass2_strict_interior_missing", "icientsxyz", "Sole missing 4-byte interior [1:5]='cien' blocks 'icients'"),
    ("pass2_strict_interior_missing", "isticalxyz", "Sole missing 4-byte interior [1:5]='stic' blocks 'istical'"),
    ("pass2_strict_interior_missing", "adressexyz", "Sole missing 4-byte interior [1:5]='dres' blocks 'adresse'"),
    ("pass2_strict_interior_missing", "ierendenxyz", "Sole missing 5-byte interior [1:6]='erend' blocks 'ierenden'"),
    ("pass2_strict_interior_missing", "😀😃", "4-byte emoji blocked by missing middle bytes \\x9f\\x98"),
    ("pass2_last_byte_ignored", "oqxyz", "2-byte 'oq' NOT in Qwen -> allowed (last byte skipped)"),
    ("pass2_last_byte_ignored", "poqxyz", "3-byte 'poq': 'oq','poq' NOT in Qwen -> allowed"),
    ("pass2_last_byte_ignored", "avoqxyz", "4-byte 'avoq': 'oq','voq','avoq' NOT in Qwen -> allowed"),
    ("pass2_last_byte_ignored", "KEYxyz", "3-byte 'KEY': [1:3]='EY' NOT in Qwen -> allowed"),
    ("pass2_last_byte_ignored", "aluexyz", "4-byte 'alue': [1:4]='lue' NOT in Qwen -> allowed"),
    ("pass2_last_byte_ignored", "ondayxyz", "5-byte 'onday': 'nday','onday' NOT in Qwen -> allowed"),
    ("pass2_last_byte_ignored", "scribxyz", "Right child 'crib' NOT in Qwen -> allowed"),
    ("pass2_last_byte_ignored", "girlsxyz", "Right child 'irls' NOT in Qwen -> allowed"),
    ("pass2_last_byte_ignored", "asoqxyz", "Right child 'oq' NOT in Qwen -> allowed"),
    ("pass2_last_byte_ignored", "abab", "Not in Qwen; all substrings of 'aba' in Qwen -> 1"),
    ("pass2_o200k_only", "olum", "In Qwen & sub_ok, NOT in o200k -> does not merge (2)"),
    ("pass2_o200k_only", "olumn", "In Qwen & sub_ok, NOT in o200k -> does not merge (2)"),
    ("pass2_o200k_rank_order", "GMMEA", "o200k rank order (3) vs Qwen rank / longest (2)"),
    ("pass2_o200k_rank_order", "oaudel", "o200k rank order (2) vs Qwen rank (3)"),
    ("pass2_o200k_rank_order", "romehen", "o200k rank order (2) vs Qwen rank (3)"),
    ("pass2_o200k_rank_order", "Latetud", "o200k rank order (2) vs longest-first (3)"),
    ("pass2_o200k_rank_order", "ohicji", "o200k rank order (3) vs longest-first (2)"),
    ("pass2_o200k_rank_order", "Locksviz", "o200k rank order (3) vs longest-first (2)"),
    ("pass2_qwen35_extended_vocab", "ationxyz", "Needs Qwen 3.5 'tio' (ID 179,293 > 151,936)"),
    ("pass2_qwen35_extended_vocab", "tringxyz", "Needs Qwen 3.5 'trin' (ID 224,604 > 151,936)"),
    ("pass2_qwen35_extended_vocab", "lementsxyz", "7-byte token with all substrings of 'lement' in Qwen -> 2"),
    ("pass2_locality", "Javaonday", "Blocked 'Java' does not prevent 'onday' in same chunk -> 3"),
    ("pass2_locality", "ondayJava", "Allowed 'onday' does not unblock 'Java' in same chunk -> 3"),
    ("workload_distortion", "The quick brown fox jumps over the lazy dog", "English prose: +0.0% vs Qwen 3.5"),
    ("workload_distortion", "userProfileController", "CamelCase: +100% vs Qwen 3.5"),
    ("workload_distortion", "Привет мир модели машинного обучения", "Russian prose: +125% vs Qwen 3.5"),
    ("workload_distortion", "人工智能模型在处理文本时会将字符串分割为词元序列", "Chinese prose: +100% vs Qwen 3.5"),
]


def main() -> None:
    api_key = get_api_key()
    jev_tok = JevBillingTokenizer()
    qwen_tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3.5-9B", "tokenizer.json"))
    o200k = jev_tok.o200k

    # Bracket every probe between "###\n" and " x\n###" so trailing punctuation
    # never merges with "\n", and subtract the baseline "###\nx x\n###" minus 1.
    base_resp = query_jev(api_key, state="###\nx x\n###")
    resolved_model = base_resp["model"]
    base_tokens = base_resp["usage"]["input_tokens"]

    def run_state_probe(item: tuple[str, str, str]) -> dict:
        cat, text, note = item
        resp = query_jev(api_key, state="###\n" + text + " x\n###")
        live = resp["usage"]["input_tokens"] - base_tokens + 1
        pred_pieces = jev_tok.encode_pieces(text)
        pred = len(pred_pieces)
        qwen_count = len(qwen_tok.encode(text, add_special_tokens=False).ids)
        o200k_count = len(o200k.encode(text, disallowed_special=()))
        return {
            "category": cat,
            "text": text,
            "live_jev": live,
            "predicted_jev": pred,
            "match": live == pred,
            "qwen35": qwen_count,
            "o200k": o200k_count,
            "predicted_pieces_repr": [repr(p) for p in pred_pieces],
            "note": note,
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        state_results = list(pool.map(run_state_probe, STATE_PROBES))

    # Verify output_tokens uses the exact same tokenizer on question IDs (never sent to GPU).
    out_base = query_jev(api_key, state="hi", instructions="ok?", question_id="x")
    out_base_tokens = out_base["usage"]["output_tokens"]
    out_base_in = out_base["usage"]["input_tokens"]
    out_qids = [
        "hello",
        "helloworld",
        "Java",
        "JavaScript",
        "oworld",
        "inquxyz",
        "errorsxyz",
        "icientsxyz",
        "ondayxyz",
        "yarakat",
        "123456",
    ]

    def run_out_probe(qid: str) -> dict:
        r = query_jev(api_key, state="hi", instructions="ok?", question_id=qid)
        live_out = r["usage"]["output_tokens"] - out_base_tokens + 1
        pred = jev_tok.count(qid)
        return {
            "question_id": qid,
            "input_tokens_delta": r["usage"]["input_tokens"] - out_base_in,
            "live_output_tokens": live_out,
            "predicted_jev": pred,
            "match": live_out == pred,
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        output_results = list(pool.map(run_out_probe, out_qids))

    # Verify instructions field + version parity across jev-1.13 and jev-1.14.
    instr_base = query_jev(api_key, state="x", instructions="###\nx x\n###")["usage"]["input_tokens"]
    instr_js = query_jev(api_key, state="x", instructions="###\nJavaScript x\n###")["usage"]["input_tokens"]

    version_checks = []
    for model_slug in ("typesafe/jev-1.13", "typesafe/jev-1.14"):
        r_base = query_jev(api_key, state="###\nx x\n###", model=model_slug)
        r_js = query_jev(api_key, state="###\nJavaScript x\n###", model=model_slug)
        r_inqu = query_jev(api_key, state="###\ninquxyz x\n###", model=model_slug)
        r_ic = query_jev(api_key, state="###\nicientsxyz x\n###", model=model_slug)
        b = r_base["usage"]["input_tokens"]
        version_checks.append(
            {
                "requested_model": model_slug,
                "resolved_model": r_base["model"],
                "JavaScript": r_js["usage"]["input_tokens"] - b + 1,
                "inquxyz": r_inqu["usage"]["input_tokens"] - b + 1,
                "icientsxyz": r_ic["usage"]["input_tokens"] - b + 1,
                "all_match": (
                    r_js["usage"]["input_tokens"] - b + 1 == 5
                    and r_inqu["usage"]["input_tokens"] - b + 1 == 3
                    and r_ic["usage"]["input_tokens"] - b + 1 == 3
                ),
            }
        )

    state_mismatches = [r for r in state_results if not r["match"]]
    output_mismatches = [r for r in output_results if not r["match"]]

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": OPENROUTER_DECISIONS_URL,
        "resolved_latest_model": resolved_model,
        "summary": {
            "state_probes_total": len(state_results),
            "state_probes_matched": len(state_results) - len(state_mismatches),
            "output_token_probes_total": len(output_results),
            "output_token_probes_matched": len(output_results) - len(output_mismatches),
            "instructions_JavaScript_tokens": instr_js - instr_base + 1,
            "version_checks_all_match": all(v["all_match"] for v in version_checks),
        },
        "state_probes": state_results,
        "output_token_probes": output_results,
        "version_checks": version_checks,
    }

    out_path = ROOT / "evidence" / "live_probes.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    assert not state_mismatches, f"State mismatches: {state_mismatches}"
    assert not output_mismatches, f"Output mismatches: {output_mismatches}"
    assert report["summary"]["instructions_JavaScript_tokens"] == 5
    assert report["summary"]["version_checks_all_match"]
    print("ALL LIVE PROBES MATCHED 100%.")


if __name__ == "__main__":
    main()
