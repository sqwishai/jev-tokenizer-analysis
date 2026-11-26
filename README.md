# Reverse-Engineering TypeSafe Jev's Billing Tokenizer

This repository documents the exact closed-form algorithm behind TypeSafe Jev's (`jev-1.13` and `jev-1.14`) metered token counter (`usage.input_tokens`, `usage.output_tokens`, and the 32,736 per-question context limit), proves it bit-for-bit across all 199,998 tokens of `o200k_base` and against live OpenRouter Jev probes, and gives the Occam's razor explanation for why this bizarre hybrid rule exists.

## Executive Summary (Occam's Razor)

Jev's billing tokenizer is **neither a stock-library misconfiguration nor naive overbilling**. It is a **custom API-gateway hybrid of Qwen 3.5 and OpenAI's `o200k_base` containing two stacked bugs in its fallback BPE filter**:

1. **Deliberate layer (why both Qwen 3.5 and `o200k_base` are loaded together):**
   Jev's API gateway runs a bespoke billing counter separate from the GPU model (it also counts `usage.output_tokens` on serialized response JSON containing question IDs never sent to the model). Rather than exposing Qwen 3.5's exact token counts (which immediately fingerprints the base model) or substituting plain `o200k_base` (which groups 3 digits per token vs Qwen's single-digit splitting and omits NFC normalization), the gateway combines:
   - Qwen 3.5's `NFC` normalizer and regex pre-tokenizer,
   - Qwen 3.5's reachable byte-token set `Q = bytes_256 ∪ {a + b : (a, b) ∈ qwen35_merges}` (`247,843` tokens, excluding the 201 merge-less entries in Qwen 3.5's `vocab`), and
   - `o200k_base`'s `199,998` mergeable byte ranks `O`.

2. **Accidental layer (why fallback BPE has the bizarre `cand[:-1]` all-substrings rule):**
   - **Pass 1 (whole-pretoken fast path, `tiktoken`-style):** if `piece in O and piece in Q`, emit 1 token.
   - **Pass 2 (fallback BPE, `o200k_base` rank order):** when filtering which `o200k_base` merges are allowed inside Qwen 3.5's vocabulary, the implementation makes two classic mistakes in three lines of code:
     1. **Conceptual mistake:** instead of checking BPE merge parents `(left, right)`, it checks whether **all contiguous byte substrings** of the candidate `cand` exist in `Q`.
     2. **Off-by-one mistake:** the inner slice upper bound uses `< len(cand)` instead of `<= len(cand)` (`range(i + 1, len(cand))` in Python, `(i + 1)..cand.len()` in Rust, `j < cand.length` in JS/Go):

```python
def allowed_merge(cand: bytes) -> bool:
    if cand not in O:
        return False
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):   # <-- off-by-one: skips every slice with j == len(cand)
            if cand[i:j] not in Q:
                return False
    return True
```

That single missing `+ 1` / `=` explains every seemingly absurd symptom simultaneously:
- **Why the last byte is set aside:** `j` stops at `len(cand) - 1`, skipping every slice `[i : len(cand)]` touching the final byte.
- **Why 10,551 non-Qwen `o200k_base` tokens leak into `FORMED`:** because `cand[0 : len(cand)] = cand` itself is never checked, non-Qwen tokens like `abab`, `crip`, `onday`, `oq`, `poq`, `avoq`, and *every* 2-byte `o200k_base` token merge freely, even though Pass 1 properly rejects non-Qwen whole chunks like `oworld` (no slice loop).
- **Why multi-byte UTF-8, CamelCase, and interior substrings explode:** ` tř` (`20 74 c5 99`) fails because the broken UTF-8 slice `b" t\xc5"` is not in `Q`; `JavaScript` shreds into `J|ava|S|crip|t` because `Jav` and `crip` are not in `Q`; `icients` inside `icientsxyz` fails solely because the 4-byte interior slice `cien` is not in `Q`.
- **Why English-centric QA missed it:** on ordinary English prose, Pass 1 (`piece in O and piece in Q`) intercepts almost every word and bypasses Pass 2 completely (**+0.0% error on English prose**).

---

## Verification Evidence

### 1. Full 199,998-Token Bit-for-Bit Proof (`evidence/full_vocab_verification.json`)

[`LabGuy94/jevtok`](https://github.com/LabGuy94/jevtok) empirically probed all 199,998 `o200k_base` tokens against live Jev (~600,000 API calls) and published the exact `LOOKUP` (`144,710` tokens) and `FORMED` (`53,820` tokens) bitmaps (`jev-1.13.bin`), concluding that no tested simple rule explained both sets.

Comparing our 15-line closed-form rule (`jev_billing_tokenizer.py`) against `jev-1.13.bin`:

| Check across `o200k_base` (199,998 tokens) | Result |
|---|---:|
| `LOOKUP` discrepancies (all valid NFC single pretokens, 144,710 tokens) | **0** |
| `FORMED` discrepancies (all valid NFC UTF-8 tokens, 53,820 tokens) | **0** |
| Multi-byte invalid-UTF-8 tokens (`840` present in both, `391` absent in both, `203` unobservable/inert in `jev-1.13.bin`) | **0 only in reconstructed** |
| Merge-less Qwen 3.5 vocab entries (`o200k ∩ (q_vocab \ q_merges)`, 62 tokens) in `LOOKUP` | **0 / 62** |
| Non-Qwen `o200k_base` tokens leaked into `FORMED` via off-by-one | **10,551** |
| Shared `o200k ∩ Qwen3.5` tokens blocked in Pass 2 | **91,846** |

Breakdown of single-missing-substring exclusions in `o200k_base` (all confirmed excluded in `jev-1.13.bin`):
- **7,995** tokens excluded *solely* because `cand[:-1]` (`[0 : L-1]`) is missing from `Q` (e.g., `Java` -> `Jav`, `orld` -> `orl`)
- **1,122** tokens excluded *solely* because a strict prefix `[0 : k]` (`k < L-1`) is missing (e.g., `Btn` -> `Bt`, `create` -> `crea`, `thing` -> `thi`, `Event` -> `Eve`)
- **6,167** tokens excluded *solely* because a strict suffix `[k : L-1]` (`k > 0`) of `cand[:-1]` is missing (e.g., `inqu` -> `nq`, `order` -> `rde`, ` minul` -> `minu`)
- **472** tokens excluded *solely* because a **strictly interior** substring `[a : b]` (`0 < a < b < L-1`) is missing (e.g., `errors` -> `rro`, `icients` -> `cien`, `istical` -> `stic`, `adresse` -> `dres`, `ierenden` -> `erend`)

### 2. Live Discriminating Probes (`evidence/live_probes.json`)

Every probe class was verified live against OpenRouter (`POST https://openrouter.ai/api/alpha/decisions`, `~typesafe/jev-latest` resolving to `typesafe/jev-1.14-20261014`, plus version parity checks on `typesafe/jev-1.13-20260917`), achieving **100% match across all state probes, `output_tokens` probes, `instructions` probes, and both model versions**:

| Probe category | Representative strings | Live Jev | Predicted | Qwen 3.5 | `o200k` | What it isolates |
|---|---|---:|---:|---:|---:|---|
| Normalization & regex | `123456`, `don't won't I've we're`, `cafe\u0301`, `\ufb01le` | `6, 8, 2, 3` | `6, 8, 2, 3` | `6, 8, 2, 2` | `2, 4, 3, 2` | Qwen 3.5 `NFC` + `\p{N}` + contraction regex |
| Special tokens | `a <\|endoftext\|> b`, `a <\|im_start\|> b` | `9, 8` | `9, 8` | `4, 4` | `9, 8` | Special tokens treated as plain text |
| Pass 1 reachable check | `yarakat`, ` terdapat`, ` dikkat`, ` факт`, `матри`, `加拿大` | `2, 2, 2, 4, 5, 3` | `2, 2, 2, 4, 5, 3` | `3` each | `1` each | Pass 1 checks `q_merges` (reachable tokens), not `q_vocab` |
| Pass 1 non-Qwen reject | `oworld` | `3` | `3` | `2` | `1` | Pass 1 requires Qwen membership (`ow\|or\|ld`) |
| Strictly interior missing | `inquxyz` (`nq`), `errorsxyz` (`rro`), `icientsxyz` (`cien`), `ierendenxyz` (`erend`), `😀😃` (`\x9f\x98`) | `3, 3, 3, 3, 4` | `3, 3, 3, 3, 4` | `2, 2, 2, 2, 3` | `2, 2, 2, 2, 3` | Any missing interior slice `[a:b]` (`b < L`) blocks merge |
| Last-byte slices ignored | `oqxyz`, `poqxyz`, `avoqxyz`, `KEYxyz`, `aluexyz`, `ondayxyz`, `scribxyz`, `girlsxyz` | `2` each | `2` each | `2–3` | `2` each | Missing `[k:L]` touching final byte never blocks merge |
| Merge rank priority | `GMMEA`, `oaudel`, `romehen`, `Latetud`, `ohicji`, `Locksviz` | `3, 2, 2, 3, 3, 3` | `3, 2, 2, 3, 3, 3` | — | — | Strictly `o200k_base` rank order |
| Qwen 3.5 extended vocab | `ationxyz` (`tio` ID 179,293), `tringxyz` (`trin` ID 224,604), `pload` (ID 192,153) | `2, 2, 1` | `2, 2, 1` | `2, 2, 1` | `2, 2, 1` | Full 248k Qwen 3.5 vocab (not Qwen 2.5/3 151k) |
| `output_tokens` & `instructions` | Question IDs `JavaScript`, `inquxyz`, `icientsxyz`, `yarakat` | `5, 3, 3, 2` | `5, 3, 3, 2` | — | — | Same gateway counter used on serialized response JSON |

---

## Why Simpler Explanations Are Mathematically Ruled Out

1. **Not a sequential two-tokenizer pipeline (Qwen 3.5 -> `o200k` or `o200k` -> Qwen 3.5):**
   On `inquxyz`, `errorsxyz`, `icientsxyz`, and `ierendenxyz`, *both* Qwen 3.5 and `o200k_base` emit 2 tokens, yet Jev emits 3. Conversely, on `yarakat`, ` terdapat`, and ` dikkat`, Qwen 3.5 emits 3 tokens (`y|arak|at`) while Jev emits 2 (`yar|akat`) across Qwen's boundaries.
2. **Not stock `tiktoken` with any single `mergeable_ranks` dict:**
   `Java`, `inqu`, `errors`, `icients`, and `ierenden` count as 1 token standalone (Pass 1), yet the exact same adjacent pieces (`J+ava`, `in+qu`, `err+ors`, `ici+ents`) refuse to merge inside a longer pretoken (Pass 2). Standard `tiktoken` uses one dictionary for both paths.
3. **Not stock HuggingFace `tokenizers` with swapped `vocab` / `merges`:**
   `10,551` tokens in `FORMED` (`abab`, `crip`, `onday`, `oq`, `avoq`, ...) do not exist in Qwen 3.5's vocabulary, so HF BPE with `vocab = Qwen3.5` cannot merge them; meanwhile `inqu`/`errors`/`icients` exist with identical splits in both tokenizers and would merge.
4. **Not a `byte_pair_merge` indexing bug, prefix-trie bug, or Aho-Corasick bug (query-complexity lower bound):**
   For `icientsxyz` (`ici + ents`), `errorsxyz` (`err + ors`), and `ierendenxyz`, every node in *both* `o200k`'s and Qwen 3.5's BPE merge trees exists in both vocabularies. The blocking interior slices (`cien`, `rro`, `erend`) cross early merge boundaries (`ic` rank 292, `er` rank 259, `en` rank 262) so `parts` in `byte_pair_merge` never contains their endpoints. Furthermore, `cien`, `rro`, `erend`, `Jav`, `orl`, `crip`, `stic`, `dres`, `minu`, `eig`, and `ndi` are all valid prefixes of other Qwen 3.5 tokens (`cient`, `rror`, `erende`, `Java`, `orld`, `cript`, ...), ruling out prefix-trie traversal failures. Because `FORMED` equals the logical AND over all $\binom{L}{2}$ substrings `cand[i:j]` ($0 \le i < j < L$), any reproducing code must check all $\binom{L}{2}$ slices against `Q`.
5. **Not naive deliberate overbilling:**
   Repetition itself does not bias counts — repeating any string `N` times simply scales its per-chunk ratio by `N`. Depending on content, the bug both **underbills** and **overbills** relative to Qwen 3.5:
   - Plain English prose: **+0.0%** (`9` vs `9` live; `56` vs `56` paragraph; `+0.4%` on a 733-token Markdown report)
   - Python source code: **-0.7% to -3.7% (underbills)** because `o200k` punctuation/newline tokens like `):\n` and `)]\n` leak through the off-by-one bug
   - Repeated `abab...`: **-45.0% (underbills)** because `abab` leaks through Pass 2 while Qwen 3.5 splits `ab|ab`
   - Czech (`Příliš žlutý člun minul třetí bóji.`): **+35.7%** (`19` vs `14`)
   - CamelCase (`userProfileController`, `JavaScript`): **+100% to +400%** (`6` vs `3`, `5` vs `1`)
   - Chinese / Russian prose: **+100% to +125%** (`24` vs `12`, `18` vs `8`)

---

## Reproducing Locally

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/). `Qwen/Qwen3.5-9B`'s `tokenizer.json` is fetched automatically via `huggingface_hub` (or read from the local HF cache).

```bash
# 1. Verify bit-for-bit across all 199,998 o200k_base tokens vs LabGuy94/jevtok jev-1.13.bin
uv run python scripts/verify_full_vocab.py

# 2. Re-run all live discriminating probes against OpenRouter Jev (~$0.001 total)
export OPENROUTER_API_KEY=...
uv run python scripts/run_live_probes.py
```

## Files

- `jev_billing_tokenizer.py` — Exact standalone implementation of the reconstructed billing tokenizer.
- `scripts/verify_full_vocab.py` — Reproduces the 199,998-token diff against `LabGuy94/jevtok` (`jev-1.13.bin`).
- `scripts/run_live_probes.py` — Runs all 71 live state probes, 11 `output_tokens` probes, `instructions` check, and `jev-1.13` / `jev-1.14` parity checks.
- `evidence/full_vocab_verification.json` — Verified output from `scripts/verify_full_vocab.py`.
- `evidence/live_probes.json` — Verified output from `scripts/run_live_probes.py`.
