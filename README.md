# Reverse-Engineering TypeSafe Jev's Billing Tokenizer (Text & Images)

This repository documents the exact closed-form algorithms behind TypeSafe Jev's (`jev-1.13` and `jev-1.14`) metered token counter — for **text**, **images**, **request framing constants**, and **`output_tokens`** — verifies them bit-for-bit across all 199,998 tokens of `o200k_base` and against both **direct first-party TypeSafe (`api.typesafe.ai/v1/systemone`)** and **OpenRouter (`openrouter.ai/api/alpha/decisions`)**, and benchmarks real open-source codebases, real Wikipedia articles across 16 languages, and extremal single-token multipliers.

---

## Executive Summary (Occam's Razor)

Jev's billing counter runs in **TypeSafe's own first-party API gateway** (100% identical between direct `api.typesafe.ai` and OpenRouter, and 100% identical between `jev-1.13-20260917` and `jev-1.14-20261014` on all shared features). It is **not** any stock model tokenizer in the market:

1. **Text billing: a deliberate `Qwen 3.5` + `o200k_base` hybrid with two stacked bugs in fallback BPE.**
   - **Deliberate layer:** combines Qwen 3.5–3.8's `NFC` normalizer, regex pre-tokenizer, and reachable byte-token set `Q = bytes_256 ∪ {a + b : (a, b) ∈ qwen35_merges}` (`247,843` tokens, excluding the 201 merge-less entries in Qwen 3.5's `vocab`) with OpenAI `o200k_base`'s `199,998` merge ranks `O`.
   - **Pass 1 (whole-pretoken fast path, `tiktoken`-style):** if `piece in O and piece in Q`, emit 1 token.
   - **Pass 2 (fallback BPE, `o200k_base` rank order):** contains two stacked bugs in three lines of code — checking **all contiguous byte substrings** instead of BPE merge parents, plus a classic **off-by-one (`j < len(cand)` instead of `j <= len(cand)`)**:

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

   That single missing `+ 1` / `=` simultaneously explains:
   - Why the final byte is ignored (`j` stops at `len(cand) - 1`).
   - Why **10,551 non-Qwen `o200k_base` tokens** (`abab`, `crip`, `onday`, `oq`, `):\n`, etc.) leak into `FORMED` even though Pass 1 rejects non-Qwen chunks like `oworld`.
   - Why **91,846 shared `o200k ∩ Qwen 3.5` tokens** are blocked inside longer pre-tokens, punishing **non-Latin Wikipedia articles (+47% to +267%)** and **real CamelCase source code (+19% to +32%)**, while leaving **English Wikipedia at +1.6%** and **`snake_case` code (Rust, Python) at -1.6% to -2.4%**.

2. **Image billing: `28×28` `ceil` patches × `2×` (~2.4–2.8× Qwen 3.5-VL native) + a hidden `+ T(asset_key)` bug.**
   - Exact formula: `2 + T(asset_key) + 2 * ceil(width / 28) * ceil(height / 28)` with hard dimension bounds `280 <= width, height <= 1568 px` (`10×28` to `56×28`, HTTP 400 outside).
   - Qwen 3.5-VL natively uses `32×32` patches (`patch_size=16, merge_size=2`) at `1×`, so Jev bills **2.43× to 2.81× more visual tokens** than Qwen 3.5-VL's sequence length (`1024×1024`: `2,740` billed vs `1,026` Qwen 3.5-VL; `1568×1568`: `6,274` billed vs `2,403` Qwen 3.5-VL).
   - **The asset reference key string itself is still billed as text tokens (`+ T(asset_key)`)** even though it is replaced by the image — turning TypeSafe's recommended UUID keys (and OpenRouter's inline `image_url` auto-UUIDs) into a silent **+32 to +35 extra billed tokens per image**.

---

## Real-World Workload Benchmarks (No Synthetic Cherry-Picking)

All **24 / 24** public corpora below (8 real open-source source files + 16 real Wikipedia `"Earth"` articles) were evaluated locally with our bit-exact reconstruction and **spot-checked live (first 1,500 chars of every file/article) against `typesafe/jev-1.14` with 24 / 24 exact token matches** (`evidence/workload_benchmarks.json`).

### 1. Real Wikipedia Articles (first 8,000 chars of `"Earth"` across 16 languages)

| Wikipedia article | Qwen 3.5 | `o200k` | Jev | **Jev / Qwen 3.5** | **vs. `o200k`** | Live spotcheck |
|---|---:|---:|---:|---:|---:|---|
| **Thai** (`โลก`) | 288 | 448 | **1,056** | **3.67× (+266.7%)** | +135.7% | exact |
| **Russian** (`Земля`) | 2,119 | 2,143 | **5,327** | **2.51× (+151.4%)** | +148.6% | exact |
| **Ukrainian** (`Земля`) | 2,933 | 2,852 | **6,251** | **2.13× (+113.1%)** | +119.2% | exact |
| **Arabic** (`الأرض`) | 2,413 | 2,424 | **4,786** | **1.98× (+98.3%)** | +97.4% | exact |
| **Greek** (`Γη`) | 3,138 | 3,016 | **5,748** | **1.83× (+83.2%)** | +90.6% | exact |
| **Japanese** (`地球`) | 5,033 | 6,490 | **7,937** | **1.58× (+57.7%)** | +22.3% | exact |
| **Chinese** (`地球`) | 5,263 | 6,951 | **8,177** | **1.55× (+55.4%)** | +17.6% | exact |
| **Hebrew** (`כדור הארץ`) | 4,096 | 3,141 | **6,281** | **1.53× (+53.3%)** | +100.0% | exact |
| **Hindi** (`पृथ्वी`) | 3,673 | 2,754 | **5,388** | **1.47× (+46.7%)** | +95.6% | exact |
| **Korean** (`지구`) | 4,433 | 5,097 | **6,079** | **1.37× (+37.1%)** | +19.3% | exact |
| **Vietnamese** (`Trái Đất`) | 2,066 | 2,349 | **2,612** | **1.26× (+26.4%)** | +11.2% | exact |
| **Czech** (`Země`) | 2,647 | 2,662 | **3,316** | **1.25× (+25.3%)** | +24.6% | exact |
| **Polish** (`Ziemia`) | 2,507 | 2,662 | **3,065** | **1.22× (+22.3%)** | +15.1% | exact |
| **Turkish** (`Dünya`) | 2,495 | 2,381 | **3,037** | **1.22× (+21.7%)** | +27.6% | exact |
| **German** (`Erde`) | 2,133 | 2,025 | **2,287** | **1.07× (+7.2%)** | +12.9% | exact |
| **English** (`Earth`) | 1,756 | 1,685 | **1,784** | **1.02× (+1.6%)** | +5.9% | exact |

### 2. Real Open-Source Production Code (first 30 KB of each file)

Notice the sharp split between **CamelCase/PascalCase languages (+19% to +32%)** — where Qwen's regex keeps multi-word identifiers unsplit so they miss Pass 1 and get shredded in Pass 2 — and **`snake_case` languages (-1.6% to -2.4%)**, where `_` separates words into Pass-1 hits while `o200k` punctuation merges (`):\n`, `],\n`) leak through Pass 2:

| Open-source file (first 30 KB) | Qwen 3.5 | `o200k` | Jev | **Jev / Qwen 3.5** | **vs. `o200k`** | Live spotcheck |
|---|---:|---:|---:|---:|---:|---|
| **C# / .NET** (`dotnet/runtime` `HttpClient.cs`) | 5,709 | 5,492 | **7,508** | **1.31× (+31.5%)** | +36.7% | exact |
| **TypeScript** (`microsoft/vscode` `codeEditorWidget.ts`) | 6,938 | 6,610 | **9,038** | **1.30× (+30.3%)** | +36.7% | exact |
| **Objective-C** (`AFNetworking` `AFHTTPSessionManager.m`) | 3,200 | 3,123 | **4,142** | **1.29× (+29.4%)** | +32.6% | exact |
| **Objective-C** (`SDWebImage` `SDWebImageManager.m`) | 6,502 | 6,179 | **7,830** | **1.20× (+20.4%)** | +26.7% | exact |
| **React / JS** (`facebook/react` `ReactFiberWorkLoop.js`) | 7,755 | 6,845 | **9,278** | **1.20× (+19.6%)** | +35.5% | exact |
| **Java / Spring** (`AbstractAutowireCapableBeanFactory.java`) | 7,106 | 6,575 | **8,445** | **1.19× (+18.8%)** | +28.4% | exact |
| **Rust** (`BurntSushi/ripgrep` `searcher/mod.rs`) | 7,288 | 6,709 | **7,170** | **0.98× (-1.6%)** | +6.9% | exact |
| **Python** (`python/cpython` `argparse.py`) | 6,738 | 6,292 | **6,574** | **0.98× (-2.4%)** | +4.5% | exact |

---

## Extremal Multipliers (Exhaustive Scan of All 247,843 Qwen 3.5 Tokens)

Scanning all **247,843** reachable Qwen 3.5 tokens and all **53,851** `FORMED` tokens (`scripts/find_extremal_multipliers.py`, **22 / 22** curated extremes live-verified against `typesafe/jev-1.14` in `evidence/extremal_multipliers.json`) shows extreme asymmetry: **up to 24.0× overbilling (16×–17× at arbitrary length)** vs. a hard floor of **0.50× (2.0× underbilling)**:

| Category | Exact string | Qwen 3.5 | `o200k` | Jev (live) | **Jev / Qwen 3.5** | **Jev / `o200k`** |
|---|---|---:|---:|---:|---:|---:|
| **Global single-token max** | `'/' * 96` *(C/Java comment divider)* | 1 | 2 | **24** | **24.0×** | 12.0× |
| **Scalable max (plain text)** | `(' ' + '/' * 70) * 10` | 10 | 20 | **170** | **17.0×** | 8.5× |
| **Scalable 16× vs. BOTH** | `'—' * 160` / `'~' * 320` / `'_' * 640` / `'/' * 960` | 10–15 | 10–15 | **160–240** | **16.0×** | **16.0×** |
| **Scalable 16× vs. Qwen** | `'%' * 640` | 10 | 20 | **160** | **16.0×** | 8.0× |
| **C# / .NET token** | `.DataGridViewColumnHeadersHeightSizeMode` | 1 | 8 | **15** | **15.0×** | 1.9× |
| **Thai token ("more info")** | `ข้อมูลเพิ่มเติม` | 1 | 2 | **15** | **15.0×** | **7.5×** |
| **Thai token ("privacy")** | `ความเป็นส่วนตัว` | 1 | 4 | **15** | **15.0×** | 3.8× |
| **Chinese token (footer)** | `转载请附上原文出处链接和本声明` | 1 | 10 | **15** | **15.0×** | 1.5× |
| **Russian token ("financing")** | ` финансирования` | 1 | 2 | **14** | **14.0×** | **7.0×** |
| **Russian token ("economic")** | ` экономического` | 1 | 2 | **14** | **14.0×** | **7.0×** |
| **C block comment banner** | `'/' + '*' * 96` | 1 | 2 | **13** | **13.0×** | 6.5× |
| **Objective-C / iOS token** | ` dequeueReusableCellWithIdentifier` | 1 | 5 | **12** | **12.0×** | 2.4× |
| **Arabic token ("strategic")** | ` الاستراتيجية` | 1 | 2 | **12** | **12.0×** | **6.0×** |
| **Objective-C / iOS token** | `.translatesAutoresizingMaskIntoConstraints` | 1 | 5 | **11** | **11.0×** | 2.2× |
| **Android / Java token** | `NavigationItemSelectedListener` | 1 | 4 | **11** | **11.0×** | 2.8× |
| **Java / Android token** | ` UnsupportedOperationException` | 1 | 3 | **10** | **10.0×** | 3.3× |
| **JS / Jest token** | `toHaveBeenCalledWith` | 1 | 5 | **9** | **9.0×** | 1.8× |
| **Global underbilling floor** | `'ab' * 100` / `'-' * 15 + '\n'` | 100 / 2 | 50 / 1 | **50 / 1** | **0.50×** | 1.0× |

---

## Image Tokenization & Billing Constants (`jev-1.14` & `jev-1.13`)

Verified live in `evidence/image_and_parity_probes.json` (**22/22** resolutions & 1-pixel boundaries, **8/8** dimension bounds, **10/10** format/key/placement variants, **9/9** billing constants):

### Image formula & dimension limits

$$\text{image\_tokens}(W, H, \text{key}) = 2 + T(\text{key}) + 2 \times \left\lceil \frac{W}{28} \right\rceil \times \left\lceil \frac{H}{28} \right\rceil$$

- **Exact bounds:** `280 <= width <= 1568` and `280 <= height <= 1568` (`10×28` to `56×28`). Even 1 pixel outside (`279×280`, `280×279`, `1569×280`, `280×1569`, `1920×1080`) returns `HTTP 400 Invalid image dimensions`.
- **1-pixel step proof:** `504×512` (`18×28`) -> `687` visual tokens; `505×512` -> `725`; `532×512` (`19×28`) -> `725`; `533×512` -> `763`; `1540×280` (`55×28`) -> `1,103`; `1541×280` -> `1,123` (including 1-token key `"a"`).
- **Invariants:** identical for PNG vs JPEG, solid vs random noise, `noul` vs `box`, `state` vs `instructions`, and `jev-1.13` vs `jev-1.14`. Referencing an asset twice bills it twice.
- **Asset-key billing bug:** at `280×280` (`200` visual tokens, base `267`), `key="a"` (`T=1`) bills `470`, `key="img1"` (`T=2`) bills `471`, `key="JavaScript"` (`T=5`) bills `474`, and UUID `key="a3f1c2e8-550e-41d4-a716-446655440000"` (`T=35`) adds **+34 tokens** over `"a"`. OpenRouter's inline `image_url` shorthand auto-generates a UUID asset key + array wrapper, silently adding **+32 extra tokens** at `512×512` compared to an explicit `"a"` asset key.

| Resolution | Qwen 3.5-VL native (`32×32` + 2) | Qwen 2.5-VL / Claude (`28×28`) | Jev visual tokens (`2 + 2×⌈W/28⌉×⌈H/28⌉`) | Ratio vs. Qwen 3.5-VL |
|---|---:|---:|---:|---:|
| `280 × 280` | 83 | 100 | **202** *(+ `T(key)`)* | **2.43×** |
| `512 × 512` | 258 | 361 | **724** *(+ `T(key)`)* | **2.81×** |
| `1024 × 1024` | 1,026 | 1,369 | **2,740** *(+ `T(key)`)* | **2.67×** |
| `1568 × 1568` | 2,403 | 3,136 | **6,274** *(+ `T(key)`)* | **2.61×** |

### Exact request billing constants

$$\text{input\_tokens} = 259 + S_{\text{state}} + \sum_{q} \bigl(7 + T(\text{instructions}_q) + E_q\bigr)$$

| Component | Exact constant / formula | Live `jev-1.14` proof |
|---|---|---|
| **Fixed per-request base** | **`259`** | `1 box` (`instr=""`) = `266` (`259 + 7`) |
| **Per question** | **`+7`** `+ T(instructions)` | 1 / 2 / 3 `noul` (`instr="x"`): `267`, `275`, `283` |
| `noul` (no criteria) / `box` extra ($E_q$) | **`+0`** | `267` (`noul`, `T=1`), `266` (`box`, `T=0`) |
| `choice` extra (no descriptions) | **`+4`** `+ sum(5 + T(opt_key))` | 2 opts (`x,y`): `282`; 3 opts (`x,y,z`): `288` |
| `choice` legend (if any non-empty desc) | **`+6`** `+ sum(5 + T(desc))` | 2 opts w/ `"a","b"`: `300` (`282 + 18`) |
| `score` legend | **`+6`** `+ sum(5 + T(level))` | 2 levels: `284`; 3 levels: `290` |
| Structured `state` | root `+0`, nested container `+2`, sibling sep `+1`, dict key `T(k)+3`, nested string `T(s)+2` | `state=""` (`270`) vs `state=[""]` (`272`) |
| `output_tokens` (unbilled, `noul`) | **`4 + 15 * N_questions + sum(T(qid))`** | 1 / 2 / 3 `noul` (`a,b,c`): `20`, `36`, `52` |

### Market comparison (is any other model tokenizer like this?)

- **Text:** **No.** Zero matches across 192+ public tokenizers, and impossible for any standard trained BPE/Unigram/WordPiece model due to the Pass 1 vs Pass 2 contradiction (`inqu` = 1 standalone, `inquxyz` = 3).
- **Images:** **No exact match.** Anthropic Claude (standard tier) shares the `28×28` `ceil` grid and `1,568 px` ceiling (`56×28`) and Qwen 2.5-VL shares the `28 px` factor, but neither doubles patch counts (`2×`), hard-rejects outside `[280, 1568] px` instead of resizing, or bills `+ T(asset_key)`. Qwen 3-VL / Qwen 3.5 use `32×32` patches at `1×`; OpenAI uses `32×32` patches with `1.2×`–`1.62×` multipliers; Gemini 3/4 uses flat budgets (`280/560/1120/2240`).

---

## Core Verification Proofs

### 1. Full 199,998-Token Bit-for-Bit Proof (`evidence/full_vocab_verification.json`)

Comparing our 15-line closed-form rule (`jev_billing_tokenizer.py`) against [`LabGuy94/jevtok`](https://github.com/LabGuy94/jevtok)'s 600,000-call empirical bitmap (`jev-1.13.bin`):

| Check across `o200k_base` (199,998 tokens) | Result |
|---|---:|
| `LOOKUP` discrepancies (all valid NFC single pretokens, 144,710 tokens) | **0** |
| `FORMED` discrepancies (all valid NFC UTF-8 tokens, 53,820 tokens) | **0** |
| Multi-byte invalid-UTF-8 tokens (`840` present in both, `391` absent in both, `203` unobservable/inert in `jev-1.13.bin`) | **0 only in reconstructed** |
| Merge-less Qwen 3.5 vocab entries (`o200k ∩ (q_vocab \ q_merges)`, 62 tokens) in `LOOKUP` | **0 / 62** |
| Non-Qwen `o200k_base` tokens leaked into `FORMED` via off-by-one | **10,551** |
| Shared `o200k ∩ Qwen 3.5` tokens blocked in Pass 2 | **91,846** |

Single-missing-substring exclusions inside `o200k_base` (all confirmed excluded in `jev-1.13.bin`):
- **7,995** excluded *solely* for missing `cand[:-1]` (`[0 : L-1]`, e.g., `Java` -> `Jav`, `orld` -> `orl`)
- **1,122** excluded *solely* for missing a strict prefix (`[0 : k]`, e.g., `Btn` -> `Bt`, `create` -> `crea`, `thing` -> `thi`)
- **6,167** excluded *solely* for missing a strict suffix of `cand[:-1]` (`[k : L-1]`, e.g., `inqu` -> `nq`, `order` -> `rde`, ` minul` -> `minu`)
- **472** excluded *solely* for missing a **strictly interior** slice (`[a : b]`, `0 < a < b < L-1`, e.g., `errors` -> `rro`, `icients` -> `cien`, `istical` -> `stic`, `adresse` -> `dres`, `ierenden` -> `erend`)

### 2. Live Discriminating Probes & Endpoint/Version Parity

- **Live text probes (`evidence/live_probes.json`):** **71 / 71** state probes, **11 / 11** `output_tokens` probes, `instructions` check (`JavaScript` = 5), and `jev-1.13`/`jev-1.14` parity matched 100%.
- **Version parity (`evidence/image_and_parity_probes.json`):** `typesafe/jev-1.13-20260917` vs `typesafe/jev-1.14-20261014` matched **20 / 20** across text, framing, `output_tokens`, images, and error bounds (only difference: `box` returns `400` on `1.13` and succeeds on `1.14`).
- **Endpoint parity (`evidence/image_and_parity_probes.json`):** OpenRouter (`openrouter.ai/api/alpha/decisions`, `typesafe/jev-1.14`) vs direct first-party TypeSafe (`api.typesafe.ai/v1/systemone`, `jev-1.14.0`) matched **21 / 21** (including `box` and all image probes) to the exact single token — proving all text and image billing quirks live inside **TypeSafe's own first-party server**.

### 3. Why Simpler Text-Bug Explanations Are Mathematically Ruled Out

1. **Not a sequential two-tokenizer pipeline (Qwen 3.5 -> `o200k` or vice versa):** on `inquxyz`, `errorsxyz`, `icientsxyz`, and `ierendenxyz`, *both* Qwen 3.5 and `o200k_base` emit 2 tokens, yet Jev emits 3. Conversely, on `yarakat`, ` terdapat`, and ` dikkat`, Qwen 3.5 emits 3 tokens (`y|arak|at`) while Jev emits 2 (`yar|akat`) across Qwen's boundaries.
2. **Not stock `tiktoken` with a single `mergeable_ranks` dict:** `Java`, `inqu`, `errors`, `icients`, and `ierenden` count as 1 token standalone (Pass 1), yet the exact same adjacent pieces (`J+ava`, `in+qu`, `err+ors`, `ici+ents`) refuse to merge inside a longer pretoken (Pass 2).
3. **Not stock HuggingFace `tokenizers` with swapped `vocab`/`merges`:** `10,551` tokens in `FORMED` (`abab`, `crip`, `onday`, `oq`, `avoq`) do not exist in Qwen 3.5's vocab, while `inqu`/`errors`/`icients` exist with identical splits in both tokenizers.
4. **Not a `byte_pair_merge` indexing bug, prefix-trie bug, or Aho-Corasick bug (query-complexity lower bound):** for `icientsxyz`, `errorsxyz`, and `ierendenxyz`, every node in both `o200k`'s and Qwen 3.5's merge trees exists in both vocabularies, the blocking interior slices (`cien`, `rro`, `erend`) cross early merge boundaries so `parts` never contains their endpoints, and every missing interior slice is a valid prefix of another Qwen 3.5 token (`cient`, `rror`, `erende`). Because `FORMED` is the logical AND of all $\binom{L}{2}$ slices `cand[i:j]` ($0 \le i < j < L$), any reproducing code must check all $\binom{L}{2}$ substrings against `Q`.

---

## Reproducing Locally

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/). `Qwen/Qwen3.5-9B`'s `tokenizer.json` is fetched automatically via `huggingface_hub`.

```bash
# 1. Verify bit-for-bit across all 199,998 o200k_base tokens vs LabGuy94/jevtok jev-1.13.bin
uv run python scripts/verify_full_vocab.py

# 2. Benchmark 8 real open-source files + 16 Wikipedia articles (add OPENROUTER_API_KEY to live spotcheck all 24)
uv run python scripts/benchmark_workloads.py

# 3. Exhaustively scan all 247,843 Qwen 3.5 tokens for extremal multipliers (add OPENROUTER_API_KEY to live-verify 22 extremes)
uv run python scripts/find_extremal_multipliers.py

# 4. Re-run all 82 live discriminating text probes (~$0.001 total)
export OPENROUTER_API_KEY=...
uv run python scripts/run_live_probes.py

# 5. Re-run image tokenization, billing constants, jev-1.13/1.14 & direct TypeSafe parity probes
export OPENROUTER_API_KEY=...
export TYPESAFE_API_KEY=...   # optional; enables direct api.typesafe.ai parity checks
uv run python scripts/run_image_and_parity_probes.py
```

## Files

- `jev_billing_tokenizer.py` — Standalone implementation of Jev's text (`JevBillingTokenizer.count`) and image (`JevBillingTokenizer.count_image`) billing counter.
- `scripts/verify_full_vocab.py` — Reproduces the 199,998-token diff against `LabGuy94/jevtok` (`jev-1.13.bin`).
- `scripts/run_live_probes.py` — Runs 71 state probes, 11 `output_tokens` probes, `instructions` check, and version parity.
- `scripts/run_image_and_parity_probes.py` — Runs 22 image resolution/step probes, 8 dimension-bound probes, 10 variant/key probes, 9 constant probes, 20 `jev-1.13`/`jev-1.14` parity probes, and 21 OpenRouter vs direct TypeSafe parity probes.
- `scripts/benchmark_workloads.py` — Benchmarks 8 real open-source code files + 16 real Wikipedia `"Earth"` articles with live spotchecks.
- `scripts/find_extremal_multipliers.py` — Exhaustive 247,843-token extremal overbilling/underbilling scanner with 22 live checks.
- `evidence/full_vocab_verification.json` — Verified 199,998-token bitmap diff output.
- `evidence/live_probes.json` — Verified live text probe results.
- `evidence/image_and_parity_probes.json` — Verified live image, constant, version-parity, and endpoint-parity results.
- `evidence/workload_benchmarks.json` — Verified 24/24 real-corpus workload benchmark results.
- `evidence/extremal_multipliers.json` — Verified 22/22 extremal multiplier scan results.
