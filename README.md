# Analysis of TypeSafe Jev's Billing Tokenizer

This repository reconstructs the token counter behind TypeSafe Jev (`jev-1.13` and `jev-1.14`) for text, images, request framing, and `output_tokens`. Every finding below comes solely from public API responses (`api.typesafe.ai/v1/systemone` and `openrouter.ai/api/alpha/decisions`) and reproducible scripts; we do not have access to TypeSafe's server source code.

All scripts and JSON outputs are included under `scripts/` and `evidence/`.

---

## Summary

Independent latency measurements by [Modaic](https://modaic.dev/blog/latency-doesnt-lie#qwens-tokenizer-predicts-jevs-latency-better-than-jevs-reported-counts) across 9,984 prompts show that Qwen 3.5's tokenizer predicts Jev's actual latency better than Jev's own reported `usage.input_tokens`, alongside throughput and self-identification evidence pointing to a Qwen 3.5 backbone.

Our reconstruction of `usage.input_tokens` matches `LabGuy94/jevtok`'s 600,000-call empirical bitmap of all 199,998 `o200k_base` tokens with **0 observable mismatches**, and matches live `jev-1.13` and `jev-1.14` responses identically on both direct `api.typesafe.ai` and OpenRouter.

In our view, the simplest explanation that fits both the latency evidence and the exact token-counting behavior is:

1. **Text billing looks like a custom gateway counter designed to decouple billed counts from Qwen 3.5 (obscuring base-model fingerprinting), combined with an off-by-one bug in its fallback BPE filter.**
   - Returning raw Qwen 3.5 token counts makes the base model trivial to fingerprint from `usage.input_tokens`. Substituting plain `o200k_base` would diverge 2-3x on digits (`\p{N}{1,3}` vs Qwen's single-digit `\p{N}`) and skip `NFC` normalization, misaligning with the 32,736-token per-question limit.
   - Instead, Jev's counter combines Qwen 3.5's `NFC` normalizer, Qwen 3.5's regex pre-tokenizer, and Qwen 3.5's reachable byte-token set `Q = bytes_256 ∪ {a + b : (a, b) ∈ qwen35_merges}` (`247,843` tokens, excluding the 201 merge-less entries in Qwen 3.5's `vocab`) with OpenAI `o200k_base`'s `199,998` merge ranks `O`.
   - **Pass 1 (whole-pretoken lookup):** if `piece in O and piece in Q`, emit 1 token.
   - **Pass 2 (fallback BPE in `o200k_base` rank order):** a candidate merge `cand` is allowed if and only if `cand in O` and every non-empty contiguous byte substring of `cand[:-1]` is in `Q`. That matches a three-line filter with two apparent mistakes: checking all contiguous substrings instead of BPE merge parents, and using `< len(cand)` instead of `<= len(cand)`:

```python
def allowed_merge(cand: bytes) -> bool:
    if cand not in O:
        return False
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):   # skips every slice with j == len(cand)
            if cand[i:j] not in Q:
                return False
    return True
```

   That single missing `+ 1` explains all three observed artifacts:
   - Substrings touching the final byte are never checked (`j` stops at `len(cand) - 1`).
   - **10,551 non-Qwen `o200k_base` tokens** (`abab`, `crip`, `onday`, `oq`, `):\n`, etc.) pass Pass 2 even though Pass 1 rejects non-Qwen chunks like `oworld`.
   - **91,846 shared `o200k ∩ Qwen 3.5` tokens** are blocked inside longer chunks. On English Wikipedia, Pass 1 catches most words and keeps billed counts within **+1.6%** of Qwen 3.5, which may explain why the behavior went unnoticed in English testing. On non-Latin Wikipedia articles, billed counts run **+47% to +267%** above Qwen 3.5, and on comment-stripped CamelCase source code they run **+21% to +42%** above Qwen 3.5 (**+30% to +49%** above `o200k_base`).

2. **Image billing uses a `28x28` `ceil` grid multiplied by `2x`, plus the text token length of the asset key.**
   - Exact observed formula: `2 + T(asset_key) + 2 * ceil(width / 28) * ceil(height / 28)` for `280 <= width, height <= 1568` px (`10x28` to `56x28`; HTTP 400 outside this range).
   - Compared to Qwen 3.5-VL's native `32x32` patch grid (`patch_size=16, merge_size=2`, `1x`), this formula bills **2.43x to 2.81x** more visual tokens (`1024x1024`: `2,740` billed vs `1,026` Qwen 3.5-VL; `1568x1568`: `6,274` billed vs `2,403` Qwen 3.5-VL) and masks the `32`-pixel patch boundary.
   - Billing `+ T(asset_key)` charges text tokens for the reference key string even though the image replaces it. With UUID asset keys (or OpenRouter's inline `image_url`, which generates a UUID key), this adds **32 to 35 extra tokens per image**.

---

## Benchmarks on Public Corpora

All **28 / 28** corpora below (16 Wikipedia `"Earth"` articles + 12 open-source code files) were measured locally with `JevBillingTokenizer` and spot-checked live (first 1,500 characters of each corpus) against `typesafe/jev-1.14` with **28 / 28 exact matches** (`evidence/workload_benchmarks.json`).

### 1. Wikipedia Articles (first 8,000 chars of `"Earth"`)

| Wikipedia article | Qwen 3.5 | `o200k` | Jev | **Jev / Qwen 3.5** | **vs. `o200k`** | Live spotcheck |
|---|---:|---:|---:|---:|---:|---|
| **Thai** (`โลก`) | 288 | 448 | **1,056** | **3.67x (+266.7%)** | +135.7% | exact |
| **Russian** (`Земля`) | 2,119 | 2,143 | **5,327** | **2.51x (+151.4%)** | +148.6% | exact |
| **Ukrainian** (`Земля`) | 2,933 | 2,852 | **6,251** | **2.13x (+113.1%)** | +119.2% | exact |
| **Arabic** (`الأرض`) | 2,413 | 2,424 | **4,786** | **1.98x (+98.3%)** | +97.4% | exact |
| **Greek** (`Γη`) | 3,138 | 3,016 | **5,748** | **1.83x (+83.2%)** | +90.6% | exact |
| **Japanese** (`地球`) | 5,033 | 6,490 | **7,937** | **1.58x (+57.7%)** | +22.3% | exact |
| **Chinese** (`地球`) | 5,263 | 6,951 | **8,177** | **1.55x (+55.4%)** | +17.6% | exact |
| **Hebrew** (`כדור הארץ`) | 4,096 | 3,141 | **6,281** | **1.53x (+53.3%)** | +100.0% | exact |
| **Hindi** (`पृथ्वी`) | 3,673 | 2,754 | **5,388** | **1.47x (+46.7%)** | +95.6% | exact |
| **Korean** (`지구`) | 4,433 | 5,097 | **6,079** | **1.37x (+37.1%)** | +19.3% | exact |
| **Vietnamese** (`Trái Đất`) | 2,066 | 2,349 | **2,612** | **1.26x (+26.4%)** | +11.2% | exact |
| **Czech** (`Země`) | 2,647 | 2,662 | **3,316** | **1.25x (+25.3%)** | +24.6% | exact |
| **Polish** (`Ziemia`) | 2,507 | 2,662 | **3,065** | **1.22x (+22.3%)** | +15.1% | exact |
| **Turkish** (`Dünya`) | 2,495 | 2,381 | **3,037** | **1.22x (+21.7%)** | +27.6% | exact |
| **German** (`Erde`) | 2,133 | 2,025 | **2,287** | **1.07x (+7.2%)** | +12.9% | exact |
| **English** (`Earth`) | 1,756 | 1,685 | **1,784** | **1.02x (+1.6%)** | +5.9% | exact |

### 2. Open-Source Source Code (first 30 KB: raw vs. comments stripped)

Qwen's regex does not split CamelCase identifiers, so multi-word symbols enter Pass 2 as a single chunk and fragment when interior substrings are absent from `Q`. Stripping license headers and doc comments isolates the code ratio (+21% to +42% vs. Qwen 3.5; +30% to +49% vs. `o200k`):

| Open-source file (first 30 KB) | Raw vs. Qwen | Code-only Qwen 3.5 | Code-only `o200k` | Code-only Jev | **Code-only vs. Qwen 3.5** | **Code-only vs. `o200k`** | Live spotcheck |
|---|---:|---:|---:|---:|---:|---:|---|
| **Objective-C / iOS** (`Instagram/IGListKit` `IGListAdapter.m`) | 1.40x | 6,113 | 6,013 | **8,660** | **1.42x (+41.7%)** | **+44.0%** | exact |
| **Java / Spring** (`RequestMappingHandlerAdapter.java`) | 1.28x | 6,380 | 5,888 | **8,772** | **1.38x (+37.5%)** | **+49.0%** | exact |
| **Android Material** (`TextInputLayout.java`) | 1.27x | 6,436 | 6,168 | **8,809** | **1.37x (+36.9%)** | **+42.8%** | exact |
| **C# / .NET** (`dotnet/runtime` `HttpClient.cs`) | 1.31x | 5,559 | 5,283 | **7,606** | **1.37x (+36.8%)** | **+44.0%** | exact |
| **Android Material** (`NavigationBarView.java`) | 1.23x | 6,274 | 5,877 | **8,279** | **1.32x (+32.0%)** | **+40.9%** | exact |
| **Objective-C / iOS** (`AFNetworking` `AFHTTPSessionManager.m`) | 1.29x | 2,897 | 2,833 | **3,790** | **1.31x (+30.8%)** | **+33.8%** | exact |
| **C# / .NET WinForms** (`DataGridView.cs`) | 1.29x | 6,863 | 6,412 | **8,930** | **1.30x (+30.1%)** | **+39.3%** | exact |
| **Android Jetpack Compose / Kotlin** (`AndroidComposeView.android.kt`) | 1.26x | 6,070 | 5,834 | **7,850** | **1.29x (+29.3%)** | **+34.6%** | exact |
| **TypeScript** (`microsoft/vscode` `codeEditorWidget.ts`) | 1.30x | 7,018 | 6,676 | **9,051** | **1.29x (+29.0%)** | **+35.6%** | exact |
| **AndroidX** (`RecyclerView.java`) | 1.11x | 6,114 | 5,729 | **7,606** | **1.24x (+24.4%)** | **+32.8%** | exact |
| **Swift / iOS** (`Alamofire` `Session.swift`) | 1.15x | 5,053 | 4,605 | **6,253** | **1.24x (+23.7%)** | **+35.8%** | exact |
| **React / JS** (`facebook/react` `ReactFiberWorkLoop.js`) | 1.20x | 7,947 | 6,959 | **9,596** | **1.21x (+20.7%)** | **+37.9%** | exact |

---

## Extremal Token Multipliers

An exhaustive scan of all **247,843** reachable Qwen 3.5 tokens and all **53,851** `FORMED` tokens (`scripts/find_extremal_multipliers.py`, **22 / 22** curated cases live-verified on `typesafe/jev-1.14` in `evidence/extremal_multipliers.json`) shows a maximum ratio of **24.0x** higher than Qwen 3.5 (**16x-17x** scalable at arbitrary length) versus a floor of **0.50x** (2.0x lower):

| Category | Exact string | Qwen 3.5 | `o200k` | Jev (live) | **Jev / Qwen 3.5** | **Jev / `o200k`** |
|---|---|---:|---:|---:|---:|---:|
| Global single-token max | `'/' * 96` (C/Java divider) | 1 | 2 | **24** | **24.0x** | 12.0x |
| Scalable plain-text max | `(' ' + '/' * 70) * 10` | 10 | 20 | **170** | **17.0x** | 8.5x |
| Scalable 16x vs. both | `'\u2014' * 160` / `'~' * 320` / `'_' * 640` / `'/' * 960` | 10-15 | 10-15 | **160-240** | **16.0x** | **16.0x** |
| Scalable 16x vs. Qwen | `'%' * 640` | 10 | 20 | **160** | **16.0x** | 8.0x |
| C# / .NET token | `.DataGridViewColumnHeadersHeightSizeMode` | 1 | 8 | **15** | **15.0x** | 1.9x |
| Thai ("more info") | `ข้อมูลเพิ่มเติม` | 1 | 2 | **15** | **15.0x** | **7.5x** |
| Thai ("privacy") | `ความเป็นส่วนตัว` | 1 | 4 | **15** | **15.0x** | 3.8x |
| Chinese footer | `转载请附上原文出处链接和本声明` | 1 | 10 | **15** | **15.0x** | 1.5x |
| Russian ("financing") | ` финансирования` | 1 | 2 | **14** | **14.0x** | **7.0x** |
| Russian ("economic") | ` экономического` | 1 | 2 | **14** | **14.0x** | **7.0x** |
| C block comment header | `'/' + '*' * 96` | 1 | 2 | **13** | **13.0x** | 6.5x |
| Objective-C / iOS token | ` dequeueReusableCellWithIdentifier` | 1 | 5 | **12** | **12.0x** | 2.4x |
| Arabic ("strategic") | ` الاستراتيجية` | 1 | 2 | **12** | **12.0x** | **6.0x** |
| Objective-C / iOS token | `.translatesAutoresizingMaskIntoConstraints` | 1 | 5 | **11** | **11.0x** | 2.2x |
| Android / Java token | `NavigationItemSelectedListener` | 1 | 4 | **11** | **11.0x** | 2.8x |
| Java / Android token | ` UnsupportedOperationException` | 1 | 3 | **10** | **10.0x** | 3.3x |
| JS / Jest token | `toHaveBeenCalledWith` | 1 | 5 | **9** | **9.0x** | 1.8x |
| Global lower bound | `'ab' * 100` / `'-' * 15 + '\n'` | 100 / 2 | 50 / 1 | **50 / 1** | **0.50x** | 1.0x |

---

## Image Tokenization and Request Constants

All image and framing numbers below are verified live in `evidence/image_and_parity_probes.json` (**22/22** resolutions and 1-pixel steps, **8/8** dimension bounds, **10/10** format/key/placement variants, **9/9** framing constants).

### Image formula and bounds

$$\text{image\_tokens}(W, H, \text{key}) = 2 + T(\text{key}) + 2 \times \left\lceil \frac{W}{28} \right\rceil \times \left\lceil \frac{H}{28} \right\rceil$$

- **Bounds:** `280 <= width <= 1568` and `280 <= height <= 1568` (`10x28` to `56x28`). Images 1 pixel outside (`279x280`, `280x279`, `1569x280`, `280x1569`, `1920x1080`) return `HTTP 400 Invalid image dimensions`.
- **1-pixel step checks:** `504x512` (`18x28`) = `687` visual tokens; `505x512` = `725`; `532x512` (`19x28`) = `725`; `533x512` = `763`; `1540x280` (`55x28`) = `1,103`; `1541x280` = `1,123` (including 1-token key `"a"`).
- **Invariants:** identical across PNG and JPEG, solid color and random noise, `noul` and `box`, `state` and `instructions`, and `jev-1.13` and `jev-1.14`. Referencing an asset twice bills it twice.
- **Asset-key billing:** at `280x280` (`200` visual tokens, base `267`), `key="a"` (`T=1`) bills `470`, `key="img1"` (`T=2`) bills `471`, `key="JavaScript"` (`T=5`) bills `474`, and UUID `key="a3f1c2e8-550e-41d4-a716-446655440000"` (`T=35`) adds **+34 tokens** over `"a"`. OpenRouter's inline `image_url` rewrites the image to a generated UUID key inside a list, adding **+32 tokens** at `512x512` over an explicit `"a"` key.

| Resolution | Qwen 3.5-VL native (`32x32` + 2) | Qwen 2.5-VL / Claude (`28x28`) | Jev visual tokens (`2 + 2*ceil(W/28)*ceil(H/28)`) | Ratio vs. Qwen 3.5-VL |
|---|---:|---:|---:|---:|
| `280 x 280` | 83 | 100 | **202** *(+ `T(key)`)* | **2.43x** |
| `512 x 512` | 258 | 361 | **724** *(+ `T(key)`)* | **2.81x** |
| `1024 x 1024` | 1,026 | 1,369 | **2,740** *(+ `T(key)`)* | **2.67x** |
| `1568 x 1568` | 2,403 | 3,136 | **6,274** *(+ `T(key)`)* | **2.61x** |

### Request billing constants

$$\text{input\_tokens} = 259 + S_{\text{state}} + \sum_{q} \bigl(7 + T(\text{instructions}_q) + E_q\bigr)$$

| Component | Formula / constant | Live `jev-1.14` check |
|---|---|---|
| Per-request base | **`259`** | `1 box` (`instr=""`) = `266` (`259 + 7`) |
| Per question | **`+7`** `+ T(instructions)` | 1 / 2 / 3 `noul` (`instr="x"`): `267`, `275`, `283` |
| `noul` (no criteria) / `box` extra ($E_q$) | **`+0`** | `267` (`noul`, `T=1`), `266` (`box`, `T=0`) |
| `choice` extra (no descriptions) | **`+4`** `+ sum(5 + T(opt_key))` | 2 opts (`x,y`): `282`; 3 opts (`x,y,z`): `288` |
| `choice` legend (any non-empty desc) | **`+6`** `+ sum(5 + T(desc))` | 2 opts w/ `"a","b"`: `300` (`282 + 18`) |
| `score` legend | **`+6`** `+ sum(5 + T(level))` | 2 levels: `284`; 3 levels: `290` |
| Structured `state` | root `+0`, nested container `+2`, sibling sep `+1`, dict key `T(k)+3`, nested string `T(s)+2` | `state=""` (`270`) vs `state=[""]` (`272`) |
| `output_tokens` (unbilled, `noul`) | **`4 + 15 * N_questions + sum(T(qid))`** | 1 / 2 / 3 `noul` (`a,b,c`): `20`, `36`, `52` |

### Comparison with other public tokenizers

- **Text:** No public model tokenizer matches Jev (0 matches across 192+ public tokenizers tested in prior studies). No standard BPE, Unigram, or WordPiece tokenizer produces the Pass 1 vs. Pass 2 mismatch (`inqu` = 1 standalone, `inquxyz` = 3).
- **Images:** No exact match. Anthropic Claude (standard tier) uses `28x28` `ceil` patches with a `1,568 px` long-edge cap, and Qwen 2.5-VL uses a `28 px` factor, but neither applies a `2x` multiplier, rejects images outside `280-1,568 px` instead of resizing, or bills `+ T(asset_key)`. Qwen 3-VL and Qwen 3.5 use `32x32` patches at `1x`; OpenAI uses `32x32` patches with `1.2x`-`1.62x` multipliers; Gemini 3/4 uses fixed budgets (`280/560/1120/2240`).

---

## Verification Details

### 1. Full 199,998-token diff (`evidence/full_vocab_verification.json`)

Comparison of `jev_billing_tokenizer.py` against [`LabGuy94/jevtok`](https://github.com/LabGuy94/jevtok)'s 600,000-call bitmap (`jev-1.13.bin`):

| Check across `o200k_base` (199,998 tokens) | Result |
|---|---:|
| `LOOKUP` discrepancies (all valid NFC single pretokens, 144,710 tokens) | **0** |
| `FORMED` discrepancies (all valid NFC UTF-8 tokens, 53,820 tokens) | **0** |
| Multi-byte invalid-UTF-8 tokens (`840` present in both, `391` absent in both, `203` unobservable/inert in `jev-1.13.bin`) | **0 only in reconstructed** |
| Merge-less Qwen 3.5 vocab entries (`o200k ∩ (q_vocab \ q_merges)`, 62 tokens) in `LOOKUP` | **0 / 62** |
| Non-Qwen `o200k_base` tokens present in `FORMED` | **10,551** |
| Shared `o200k ∩ Qwen 3.5` tokens excluded from `FORMED` | **91,846** |

Single-missing-substring exclusions inside `o200k_base` (all confirmed excluded in `jev-1.13.bin`):
- **7,995** excluded solely for missing `cand[:-1]` (`[0 : L-1]`, e.g. `Java` -> `Jav`, `orld` -> `orl`)
- **1,122** excluded solely for missing a strict prefix (`[0 : k]`, e.g. `Btn` -> `Bt`, `create` -> `crea`, `thing` -> `thi`)
- **6,167** excluded solely for missing a strict suffix of `cand[:-1]` (`[k : L-1]`, e.g. `inqu` -> `nq`, `order` -> `rde`, ` minul` -> `minu`)
- **472** excluded solely for missing a strictly interior slice (`[a : b]`, `0 < a < b < L-1`, e.g. `errors` -> `rro`, `icients` -> `cien`, `istical` -> `stic`, `adresse` -> `dres`, `ierenden` -> `erend`)

### 2. Live probes and parity

- **Text probes (`evidence/live_probes.json`):** **71 / 71** state probes, **11 / 11** `output_tokens` probes, `instructions` check (`JavaScript` = 5), and `jev-1.13`/`jev-1.14` parity matched 100%.
- **Version parity (`evidence/image_and_parity_probes.json`):** `typesafe/jev-1.13-20260917` vs `typesafe/jev-1.14-20261014` matched **20 / 20** on text, framing, `output_tokens`, images, and error bounds (`box` returns `400` on `1.13` and succeeds on `1.14`).
- **Endpoint parity (`evidence/image_and_parity_probes.json`):** OpenRouter (`openrouter.ai/api/alpha/decisions`, `typesafe/jev-1.14`) vs direct `api.typesafe.ai/v1/systemone` (`jev-1.14.0`) matched **21 / 21** (including `box` and all image probes) to the exact token. This confirms the billing counter runs on TypeSafe's upstream server rather than inside OpenRouter.

### 3. Why simpler explanations are ruled out

1. **Not a sequential two-tokenizer pipeline:** on `inquxyz`, `errorsxyz`, `icientsxyz`, and `ierendenxyz`, both Qwen 3.5 and `o200k_base` emit 2 tokens, while Jev emits 3. On `yarakat`, ` terdapat`, and ` dikkat`, Qwen 3.5 emits 3 tokens (`y|arak|at`) while Jev emits 2 (`yar|akat`) across Qwen's boundaries.
2. **Not stock `tiktoken` with one `mergeable_ranks` dict:** `Java`, `inqu`, `errors`, `icients`, and `ierenden` count as 1 token standalone (Pass 1), yet the same adjacent pieces (`J+ava`, `in+qu`, `err+ors`, `ici+ents`) do not merge inside a longer pretoken (Pass 2).
3. **Not stock HuggingFace `tokenizers` with swapped `vocab`/`merges`:** `10,551` tokens in `FORMED` (`abab`, `crip`, `onday`, `oq`, `avoq`) do not exist in Qwen 3.5's vocab, while `inqu`, `errors`, and `icients` exist with identical merge splits in both tokenizers.
4. **Not a `byte_pair_merge` indexing bug or prefix-trie bug:** for `icientsxyz`, `errorsxyz`, and `ierendenxyz`, every node in both `o200k`'s and Qwen 3.5's merge trees exists in both vocabularies. The blocking interior slices (`cien`, `rro`, `erend`) cross early merge boundaries so `parts` never contains their endpoints, and each missing slice is a valid prefix of another Qwen 3.5 token (`cient`, `rror`, `erende`). Because `FORMED` is the exact logical AND over all $\binom{L}{2}$ slices `cand[i:j]` ($0 \le i < j < L$), any code that reproduces it must test all $\binom{L}{2}$ substrings against `Q`.

---

## Reproducing Locally

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/). `Qwen/Qwen3.5-9B`'s `tokenizer.json` is downloaded automatically via `huggingface_hub`.

```bash
# 1. Verify across all 199,998 o200k_base tokens vs LabGuy94/jevtok jev-1.13.bin
uv run python scripts/verify_full_vocab.py

# 2. Benchmark 12 open-source files (raw & comment-stripped) + 16 Wikipedia articles
#    (set OPENROUTER_API_KEY to live spotcheck all 28)
uv run python scripts/benchmark_workloads.py

# 3. Scan all 247,843 Qwen 3.5 tokens for extremal multipliers
#    (set OPENROUTER_API_KEY to live-verify 22 extremes)
uv run python scripts/find_extremal_multipliers.py

# 4. Run all 82 live discriminating text probes (~$0.001 total)
export OPENROUTER_API_KEY=...
uv run python scripts/run_live_probes.py

# 5. Run image tokenization, billing constants, jev-1.13/1.14 & direct TypeSafe parity probes
export OPENROUTER_API_KEY=...
export TYPESAFE_API_KEY=...   # optional; enables direct api.typesafe.ai parity checks
uv run python scripts/run_image_and_parity_probes.py
```

## Files

- `jev_billing_tokenizer.py`: standalone implementation of Jev's text (`JevBillingTokenizer.count`) and image (`JevBillingTokenizer.count_image`) token counter.
- `scripts/verify_full_vocab.py`: reproduces the 199,998-token diff against `LabGuy94/jevtok` (`jev-1.13.bin`).
- `scripts/run_live_probes.py`: runs 71 state probes, 11 `output_tokens` probes, `instructions` check, and version parity.
- `scripts/run_image_and_parity_probes.py`: runs 22 image resolution/step probes, 8 dimension-bound probes, 10 variant/key probes, 9 constant probes, 20 `jev-1.13`/`jev-1.14` parity probes, and 21 OpenRouter vs. direct TypeSafe parity probes.
- `scripts/benchmark_workloads.py`: benchmarks 12 open-source code files (raw and comment-stripped) and 16 Wikipedia `"Earth"` articles with live spotchecks.
- `scripts/find_extremal_multipliers.py`: exhaustive 247,843-token extremal multiplier scanner with 22 live checks.
- `evidence/full_vocab_verification.json`: 199,998-token bitmap diff output.
- `evidence/live_probes.json`: live text probe results.
- `evidence/image_and_parity_probes.json`: live image, constant, version-parity, and endpoint-parity results.
- `evidence/workload_benchmarks.json`: 28/28 public-corpus benchmark results.
- `evidence/extremal_multipliers.json`: 22/22 extremal multiplier scan results.
