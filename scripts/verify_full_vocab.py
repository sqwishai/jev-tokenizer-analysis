from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jev_billing_tokenizer import JevBillingTokenizer, all_substrings_before_last_byte_in

JEVTOK_BIN_URL = (
    "https://raw.githubusercontent.com/LabGuy94/jevtok/main/src/jevtok/data/jev-1.13.bin"
)


def load_jevtok_bitmaps(tok: JevBillingTokenizer, url: str = JEVTOK_BIN_URL) -> tuple[set[bytes], set[bytes], int]:
    raw = requests.get(url, timeout=30).content
    if raw[:8] != b"JEVTOK1\0":
        raise ValueError("Unexpected jevtok binary magic header")
    n_vocab = int.from_bytes(raw[10:14], "little")
    size = (n_vocab + 7) // 8
    lookup_bm = raw[14 : 14 + size]
    formed_bm = raw[14 + size : 14 + 2 * size]

    jev_lookup: set[bytes] = set()
    jev_formed: set[bytes] = set()
    for i in range(n_vocab):
        try:
            piece = tok.o200k.decode_single_token_bytes(i)
        except Exception:
            continue
        if (lookup_bm[i >> 3] >> (i & 7)) & 1:
            jev_lookup.add(piece)
        if (formed_bm[i >> 3] >> (i & 7)) & 1:
            jev_formed.add(piece)
    return jev_lookup, jev_formed, n_vocab


def is_valid_nfc_utf8(bs: bytes) -> bool:
    try:
        s = bs.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return unicodedata.normalize("NFC", s) == s


def main() -> None:
    tok = JevBillingTokenizer()
    jev_lookup, jev_formed, n_vocab = load_jevtok_bitmaps(tok)

    o_ranks = tok.o200k_ranks
    q_vocab = tok.qwen_vocab
    q_reach = tok.qwen_reachable
    q_pat = tok.pretokenizer

    def is_valid_nfc_single_pretoken(bs: bytes) -> bool:
        try:
            s = bs.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return unicodedata.normalize("NFC", s) == s and q_pat.findall(s) == [s]

    our_formed_reach = set(tok.pass2_formed_ranks)
    q_vocab_set = set(q_vocab)
    our_formed_vocab = {k for k in o_ranks if all_substrings_before_last_byte_in(k, q_vocab_set)}
    our_lookup_union = tok.pass1_lookup | our_formed_reach

    lookup_extra = [t for t in (jev_lookup - our_lookup_union) if is_valid_nfc_single_pretoken(t)]
    lookup_miss = [t for t in (our_lookup_union - jev_lookup) if is_valid_nfc_single_pretoken(t)]

    formed_extra_nfc = [t for t in (jev_formed - our_formed_reach) if is_valid_nfc_utf8(t)]

    all_invalid_utf8 = []
    for t in o_ranks:
        if len(t) >= 2:
            try:
                t.decode("utf-8")
            except UnicodeDecodeError:
                all_invalid_utf8.append(t)

    inv_both = sum(1 for t in all_invalid_utf8 if t in jev_formed and t in our_formed_reach)
    inv_neither = sum(1 for t in all_invalid_utf8 if t not in jev_formed and t not in our_formed_reach)
    inv_only_jev = sum(1 for t in all_invalid_utf8 if t in jev_formed and t not in our_formed_reach)
    inv_only_ours = sum(1 for t in all_invalid_utf8 if t not in jev_formed and t in our_formed_reach)

    dead_qwen_in_o200k = [t for t in o_ranks if t in q_vocab and t not in q_reach]
    dead_in_jev_lookup = sum(1 for t in dead_qwen_in_o200k if t in jev_lookup)
    dead_in_jev_formed = sum(1 for t in dead_qwen_in_o200k if t in jev_formed)

    sole_m1 = 0
    sole_prefix = 0
    sole_suffix_of_m1 = 0
    sole_interior = 0
    interior_examples: list[dict[str, str]] = []
    for t in o_ranks:
        if len(t) < 3 or not is_valid_nfc_single_pretoken(t):
            continue
        p = t[:-1]
        missing = [
            (a, b)
            for a in range(len(p))
            for b in range(a + 1, len(p) + 1)
            if p[a:b] not in q_reach
        ]
        if len(missing) == 1:
            a, b = missing[0]
            if a == 0 and b == len(p):
                sole_m1 += 1
            elif a == 0:
                sole_prefix += 1
            elif b == len(p):
                sole_suffix_of_m1 += 1
            else:
                sole_interior += 1
                if len(interior_examples) < 12 and t.isascii() and t.isalpha():
                    interior_examples.append(
                        {"token": t.decode("ascii"), "sole_missing_substring": p[a:b].decode("ascii")}
                    )

    formed_not_in_qwen = [t for t in jev_formed if t not in q_vocab]
    formed_not_in_qwen_by_len = {
        str(k): v for k, v in sorted(Counter(len(t) for t in formed_not_in_qwen).items())
    }

    report = {
        "source_bitmap": JEVTOK_BIN_URL,
        "o200k_n_vocab": n_vocab,
        "o200k_mergeable_ranks": len(o_ranks),
        "qwen35_vocab_size": len(q_vocab),
        "qwen35_reachable_size": len(q_reach),
        "qwen35_mergeless_vocab_entries": len(q_vocab) - len(q_reach),
        "jevtok_counts": {
            "lookup": len(jev_lookup),
            "formed": len(jev_formed),
        },
        "reconstructed_counts": {
            "pass1_o200k_intersect_qwen_reachable": len(tok.pass1_lookup),
            "pass2_formed_using_qwen_reachable": len(our_formed_reach),
            "pass2_formed_using_qwen_vocab": len(our_formed_vocab),
            "lookup_union": len(our_lookup_union),
            "shared_tokens_blocked_in_pass2": len((set(o_ranks) & set(q_vocab)) - our_formed_reach),
        },
        "discrepancies": {
            "lookup_valid_nfc_single_pretoken_extra_in_jevtok": len(lookup_extra),
            "lookup_valid_nfc_single_pretoken_missing_in_jevtok": len(lookup_miss),
            "formed_valid_nfc_utf8_extra_in_jevtok": len(formed_extra_nfc),
            "invalid_utf8_only_in_ours": inv_only_ours,
        },
        "invalid_utf8_multibyte_breakdown": {
            "total_in_o200k": len(all_invalid_utf8),
            "present_in_both": inv_both,
            "absent_in_both": inv_neither,
            "only_in_jevtok_unobservable_inert": inv_only_jev,
            "only_in_reconstructed": inv_only_ours,
        },
        "mergeless_qwen_vocab_check": {
            "total_in_o200k": len(dead_qwen_in_o200k),
            "present_in_jev_lookup": dead_in_jev_lookup,
            "present_in_jev_formed": dead_in_jev_formed,
        },
        "single_missing_substring_exclusions_in_o200k": {
            "solely_missing_tok_minus_1": sole_m1,
            "solely_missing_strict_prefix": sole_prefix,
            "solely_missing_strict_suffix_of_tok_minus_1": sole_suffix_of_m1,
            "solely_missing_strict_interior": sole_interior,
            "strict_interior_ascii_examples": interior_examples,
        },
        "non_qwen_o200k_tokens_leaked_into_formed": {
            "total": len(formed_not_in_qwen),
            "by_byte_length": formed_not_in_qwen_by_len,
        },
    }

    out_path = ROOT / "evidence" / "full_vocab_verification.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    assert report["discrepancies"]["lookup_valid_nfc_single_pretoken_extra_in_jevtok"] == 0
    assert report["discrepancies"]["lookup_valid_nfc_single_pretoken_missing_in_jevtok"] == 0
    assert report["discrepancies"]["formed_valid_nfc_utf8_extra_in_jevtok"] == 0
    assert report["discrepancies"]["invalid_utf8_only_in_ours"] == 0
    assert report["mergeless_qwen_vocab_check"]["present_in_jev_lookup"] == 0
    print("\nALL FULL-VOCABULARY ASSERTIONS PASSED WITH 0 OBSERVABLE MISMATCHES.")


if __name__ == "__main__":
    main()
