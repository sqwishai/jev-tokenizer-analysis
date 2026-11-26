from __future__ import annotations

import base64
import io
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jev_billing_tokenizer import JevBillingTokenizer

OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"


def make_img_b64(w: int, h: int, fmt: str = "PNG", noise: bool = False) -> str:
    if noise:
        rng = random.Random(42)
        raw = bytes(rng.getrandbits(8) for _ in range(w * h * 3))
        im = Image.frombytes("RGB", (w, h), raw)
    else:
        im = Image.new("RGB", (w, h), (200, 50, 50))
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def post(url: str, headers: dict[str, str], body: dict) -> dict:
    r = requests.post(url, headers=headers, json=body, timeout=60)
    if r.status_code != 200:
        return {"ok": False, "status": r.status_code, "error": r.text[:200]}
    d = r.json()
    return {
        "ok": True,
        "model": d.get("model"),
        "input_tokens": d["usage"]["input_tokens"],
        "output_tokens": d["usage"]["output_tokens"],
    }


def qwen35_native_image_tokens(w: int, h: int) -> int:
    # Qwen 3.5-VL uses patch_size=16, merge_size=2 -> factor=32, smart_resize round() + 2 vision tags.
    h_bar = max(32, round(h / 32) * 32)
    w_bar = max(32, round(w / 32) * 32)
    return 2 + (h_bar // 32) * (w_bar // 32)


def main() -> None:
    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not or_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in the environment.")
    ts_key = os.environ.get("TYPESAFE_API_KEY", "").strip()

    or_headers = {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"}
    ts_headers = (
        {
            "Authorization": f"Bearer {ts_key}",
            "Content-Type": "application/json",
            "TypeSafe-Danger-Zone-Images": "true",
        }
        if ts_key
        else None
    )

    tok = JevBillingTokenizer()
    base_noul_x = post(
        OPENROUTER_URL,
        or_headers,
        {"model": "typesafe/jev-1.14", "state": "", "questions": {"q": {"type": "noul", "instructions": "x"}}},
    )
    assert base_noul_x["ok"] and base_noul_x["input_tokens"] == 267, base_noul_x

    # 1. Image resolution & 1-pixel step boundary sweep (factor=28, ceil, 2x multiplier)
    res_cases = [
        (280, 280),
        (320, 320),
        (384, 384),
        (448, 448),
        (503, 512),
        (504, 512),
        (505, 512),
        (512, 512),
        (532, 512),
        (533, 512),
        (640, 480),
        (768, 768),
        (800, 600),
        (1024, 512),
        (1024, 1024),
        (1280, 720),
        (1536, 1536),
        (1540, 280),
        (1541, 280),
        (1568, 280),
        (280, 1568),
        (1568, 1568),
    ]

    def run_res(sz: tuple[int, int]) -> dict:
        w, h = sz
        body = {
            "model": "typesafe/jev-1.14",
            "state": "a",
            "assets": {"a": {"type": "image", "format": "png", "data_base64": make_img_b64(w, h)}},
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        }
        r = post(OPENROUTER_URL, or_headers, body)
        assert r["ok"], (w, h, r)
        billed_img = r["input_tokens"] - 267
        pred_img = tok.count_image(w, h, "a")
        q35 = qwen35_native_image_tokens(w, h)
        return {
            "width": w,
            "height": h,
            "ceil_w_28": math.ceil(w / 28),
            "ceil_h_28": math.ceil(h / 28),
            "live_image_tokens_incl_key": billed_img,
            "predicted_image_tokens_incl_key": pred_img,
            "match": billed_img == pred_img,
            "live_visual_tokens_excl_key": billed_img - 1,
            "qwen35_native_tokens": q35,
            "ratio_vs_qwen35": round((billed_img - 1) / q35, 3),
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        res_results = list(pool.map(run_res, res_cases))

    # 2. Exact dimension bounds (expect 400 outside [280, 1568] on either axis)
    bound_cases = [
        (279, 280, False),
        (280, 279, False),
        (280, 280, True),
        (1568, 280, True),
        (280, 1568, True),
        (1569, 280, False),
        (280, 1569, False),
        (1920, 1080, False),
    ]

    def run_bound(item: tuple[int, int, bool]) -> dict:
        w, h, expect_ok = item
        body = {
            "model": "typesafe/jev-1.14",
            "state": "a",
            "assets": {"a": {"type": "image", "format": "png", "data_base64": make_img_b64(w, h)}},
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        }
        r = post(OPENROUTER_URL, or_headers, body)
        return {
            "width": w,
            "height": h,
            "expected_ok": expect_ok,
            "actual_ok": r["ok"],
            "status": 200 if r["ok"] else r["status"],
            "match": r["ok"] == expect_ok,
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        bound_results = list(pool.map(run_bound, bound_cases))

    # 3. Asset key billing bug, format/content/placement invariants, and inline UUID tax
    b280 = make_img_b64(280, 280)
    b512_png = make_img_b64(512, 512, fmt="PNG", noise=False)
    b512_jpg = make_img_b64(512, 512, fmt="JPEG", noise=False)
    b512_noise = make_img_b64(512, 512, fmt="PNG", noise=True)

    variant_specs = [
        (
            "key_a_T1_280",
            {"model": "typesafe/jev-1.14", "state": "a", "assets": {"a": {"type": "image", "format": "png", "data_base64": b280}}, "questions": {"q": {"type": "noul", "instructions": "x"}}},
            267 + tok.count_image(280, 280, "a"),
        ),
        (
            "key_img1_T2_280",
            {"model": "typesafe/jev-1.14", "state": "img1", "assets": {"img1": {"type": "image", "format": "png", "data_base64": b280}}, "questions": {"q": {"type": "noul", "instructions": "x"}}},
            267 + tok.count_image(280, 280, "img1"),
        ),
        (
            "key_JavaScript_T5_280",
            {"model": "typesafe/jev-1.14", "state": "JavaScript", "assets": {"JavaScript": {"type": "image", "format": "png", "data_base64": b280}}, "questions": {"q": {"type": "noul", "instructions": "x"}}},
            267 + tok.count_image(280, 280, "JavaScript"),
        ),
        (
            "key_uuid_T35_512",
            {
                "model": "typesafe/jev-1.14",
                "state": "a3f1c2e8-550e-41d4-a716-446655440000",
                "assets": {"a3f1c2e8-550e-41d4-a716-446655440000": {"type": "image", "format": "png", "data_base64": b512_png}},
                "questions": {"q": {"type": "noul", "instructions": "x"}},
            },
            267 + tok.count_image(512, 512, "a3f1c2e8-550e-41d4-a716-446655440000"),
        ),
        (
            "512_png_solid",
            {"model": "typesafe/jev-1.14", "state": "a", "assets": {"a": {"type": "image", "format": "png", "data_base64": b512_png}}, "questions": {"q": {"type": "noul", "instructions": "x"}}},
            267 + tok.count_image(512, 512, "a"),
        ),
        (
            "512_jpg_solid",
            {"model": "typesafe/jev-1.14", "state": "a", "assets": {"a": {"type": "image", "format": "jpg", "data_base64": b512_jpg}}, "questions": {"q": {"type": "noul", "instructions": "x"}}},
            267 + tok.count_image(512, 512, "a"),
        ),
        (
            "512_png_noise",
            {"model": "typesafe/jev-1.14", "state": "a", "assets": {"a": {"type": "image", "format": "png", "data_base64": b512_noise}}, "questions": {"q": {"type": "noul", "instructions": "x"}}},
            267 + tok.count_image(512, 512, "a"),
        ),
        (
            "512_in_instructions",
            {"model": "typesafe/jev-1.14", "state": "", "assets": {"a": {"type": "image", "format": "png", "data_base64": b512_png}}, "questions": {"q": {"type": "noul", "instructions": "x a"}}},
            267 + tok.count_image(512, 512, "a"),
        ),
        (
            "512_box_question",
            {"model": "typesafe/jev-1.14", "state": "a", "assets": {"a": {"type": "image", "format": "png", "data_base64": b512_png}}, "questions": {"q": {"type": "box", "instructions": "x"}}},
            267 + tok.count_image(512, 512, "a"),
        ),
        (
            "512_referenced_twice_in_array",
            {"model": "typesafe/jev-1.14", "state": ["a", "a"], "assets": {"a": {"type": "image", "format": "png", "data_base64": b512_png}}, "questions": {"q": {"type": "noul", "instructions": "x"}}},
            267 + 5 + 2 * tok.count_image(512, 512, "a"),
        ),
    ]

    def run_var(item: tuple[str, dict, int]) -> dict:
        name, body, expected = item
        r = post(OPENROUTER_URL, or_headers, body)
        assert r["ok"], (name, r)
        return {
            "name": name,
            "live_input_tokens": r["input_tokens"],
            "expected_input_tokens": expected,
            "match": r["input_tokens"] == expected,
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        variant_results = list(pool.map(run_var, variant_specs))

    inline_512 = post(
        OPENROUTER_URL,
        or_headers,
        {
            "model": "typesafe/jev-1.14",
            "state": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b512_png}"}}],
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        },
    )
    assert inline_512["ok"]
    inline_overhead_vs_short_key = inline_512["input_tokens"] - (267 + tok.count_image(512, 512, "a"))

    # 4. Exact request framing constants on jev-1.14
    const_specs = [
        ("1_noul_instr_x", {"model": "typesafe/jev-1.14", "state": "", "questions": {"a": {"type": "noul", "instructions": "x"}}}, 267, 20),
        ("2_noul_instr_x", {"model": "typesafe/jev-1.14", "state": "", "questions": {"a": {"type": "noul", "instructions": "x"}, "b": {"type": "noul", "instructions": "x"}}}, 275, 36),
        ("3_noul_instr_x", {"model": "typesafe/jev-1.14", "state": "", "questions": {"a": {"type": "noul", "instructions": "x"}, "b": {"type": "noul", "instructions": "x"}, "c": {"type": "noul", "instructions": "x"}}}, 283, 52),
        ("1_box_empty_instr", {"model": "typesafe/jev-1.14", "state": "", "questions": {"a": {"type": "box", "instructions": ""}}}, 266, 20),
        ("1_choice_2opt_nodesc", {"model": "typesafe/jev-1.14", "state": "", "questions": {"a": {"type": "choice", "instructions": "", "criteria": {"x": None, "y": None}}}}, 282, 31),
        ("1_choice_3opt_nodesc", {"model": "typesafe/jev-1.14", "state": "", "questions": {"a": {"type": "choice", "instructions": "", "criteria": {"x": None, "y": None, "z": None}}}}, 288, 38),
        ("1_choice_2opt_desc", {"model": "typesafe/jev-1.14", "state": "", "questions": {"a": {"type": "choice", "instructions": "", "criteria": {"x": "a", "y": "b"}}}}, 300, 31),
        ("1_score_2lvl", {"model": "typesafe/jev-1.14", "state": "", "questions": {"a": {"type": "score", "instructions": "", "criteria": ["a", "b"]}}}, 284, 17),
        ("1_score_3lvl", {"model": "typesafe/jev-1.14", "state": "", "questions": {"a": {"type": "score", "instructions": "", "criteria": ["a", "b", "c"]}}}, 290, 17),
    ]
    const_results = []
    for name, body, exp_in, exp_out in const_specs:
        r = post(OPENROUTER_URL, or_headers, body)
        assert r["ok"], (name, r)
        const_results.append(
            {
                "name": name,
                "live_input_tokens": r["input_tokens"],
                "expected_input_tokens": exp_in,
                "live_output_tokens": r["output_tokens"],
                "expected_output_tokens": exp_out,
                "match": r["input_tokens"] == exp_in and r["output_tokens"] == exp_out,
            }
        )

    # 5. Version parity (jev-1.13 vs jev-1.14) and endpoint parity (OpenRouter vs direct TypeSafe)
    b504 = make_img_b64(504, 512)
    b505 = make_img_b64(505, 512)
    b1568 = make_img_b64(1568, 1568)
    b279 = make_img_b64(279, 280)
    b1569 = make_img_b64(1569, 280)

    parity_cases = [
        ("text_JavaScript", {"state": "JavaScript", "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("text_123456", {"state": "123456", "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("text_nfd_nfc", {"state": "cafe\u0301 \ufb01le", "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("text_yarakat", {"state": "yarakat", "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("text_icientsxyz", {"state": "icientsxyz", "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("text_ierendenxyz", {"state": "ierendenxyz", "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("text_avoqxyz", {"state": "avoqxyz", "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("text_GMMEA", {"state": "GMMEA", "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("text_czech", {"state": "Příliš žlutý člun minul třetí bóji.", "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("text_russian", {"state": "Привет мир модели машинного обучения", "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("frame_2noul_out", {"state": "", "questions": {"JavaScript": {"type": "noul", "instructions": "x"}, "icientsxyz": {"type": "noul", "instructions": "y"}}}),
        ("frame_choice_desc", {"state": {"k": "v", "n": [1, 2]}, "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": "one", "b": "two"}}}}),
        ("frame_score", {"state": "hi", "questions": {"q": {"type": "score", "instructions": "rate", "criteria": ["low", "mid", "high"]}}}),
        ("img_280_key_a", {"state": "a", "assets": {"a": {"type": "image", "format": "png", "data_base64": b280}}, "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("img_280_key_JS", {"state": "JavaScript", "assets": {"JavaScript": {"type": "image", "format": "png", "data_base64": b280}}, "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("img_504x512", {"state": "a", "assets": {"a": {"type": "image", "format": "png", "data_base64": b504}}, "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("img_505x512", {"state": "a", "assets": {"a": {"type": "image", "format": "png", "data_base64": b505}}, "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("img_1568x1568", {"state": "a", "assets": {"a": {"type": "image", "format": "png", "data_base64": b1568}}, "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("img_err_279x280", {"state": "a", "assets": {"a": {"type": "image", "format": "png", "data_base64": b279}}, "questions": {"q": {"type": "noul", "instructions": "x"}}}),
        ("img_err_1569x280", {"state": "a", "assets": {"a": {"type": "image", "format": "png", "data_base64": b1569}}, "questions": {"q": {"type": "noul", "instructions": "x"}}}),
    ]

    def sig(r: dict) -> list:
        return ["OK", r["input_tokens"], r["output_tokens"]] if r["ok"] else ["ERR", r["status"]]

    def run_parity(item: tuple[str, dict]) -> dict:
        name, body = item
        r13 = post(OPENROUTER_URL, or_headers, dict(body, model="typesafe/jev-1.13"))
        r14 = post(OPENROUTER_URL, or_headers, dict(body, model="typesafe/jev-1.14"))
        row: dict = {
            "name": name,
            "openrouter_jev113": sig(r13),
            "openrouter_jev114": sig(r14),
            "jev113_eq_jev114": sig(r13) == sig(r14),
        }
        if ts_headers is not None:
            rts = post(TYPESAFE_URL, ts_headers, dict(body, model="jev-1.14.0"))
            row["direct_typesafe_jev114"] = sig(rts)
            row["openrouter_eq_typesafe"] = sig(r14) == sig(rts)
        return row

    with ThreadPoolExecutor(max_workers=6) as pool:
        parity_results = list(pool.map(run_parity, parity_cases))

    box_13 = post(OPENROUTER_URL, or_headers, {"model": "typesafe/jev-1.13", "state": "hi", "questions": {"q": {"type": "box", "instructions": "find"}}})
    box_14_or = post(OPENROUTER_URL, or_headers, {"model": "typesafe/jev-1.14", "state": "hi", "questions": {"q": {"type": "box", "instructions": "find"}}})
    box_check: dict = {
        "openrouter_jev113": sig(box_13),
        "openrouter_jev114": sig(box_14_or),
    }
    if ts_headers is not None:
        box_14_ts = post(TYPESAFE_URL, ts_headers, {"model": "jev-1.14.0", "state": "hi", "questions": {"q": {"type": "box", "instructions": "find"}}})
        box_check["direct_typesafe_jev114"] = sig(box_14_ts)
        box_check["openrouter_eq_typesafe"] = sig(box_14_or) == sig(box_14_ts)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "formula": "2 + T(asset_key) + 2 * ceil(width / 28) * ceil(height / 28)",
        "dimension_bounds_px": {"min_side": 280, "max_side": 1568},
        "summary": {
            "resolution_probes_matched": sum(1 for r in res_results if r["match"]),
            "resolution_probes_total": len(res_results),
            "bound_probes_matched": sum(1 for r in bound_results if r["match"]),
            "bound_probes_total": len(bound_results),
            "variant_probes_matched": sum(1 for r in variant_results if r["match"]),
            "variant_probes_total": len(variant_results),
            "openrouter_inline_512_overhead_vs_short_key": inline_overhead_vs_short_key,
            "constant_probes_matched": sum(1 for r in const_results if r["match"]),
            "constant_probes_total": len(const_results),
            "jev113_vs_jev114_matched": sum(1 for r in parity_results if r["jev113_eq_jev114"]),
            "jev113_vs_jev114_total": len(parity_results),
            "openrouter_vs_typesafe_tested": ts_headers is not None,
            "openrouter_vs_typesafe_matched": (
                sum(1 for r in parity_results if r["openrouter_eq_typesafe"])
                + (1 if box_check.get("openrouter_eq_typesafe") else 0)
                if ts_headers is not None
                else None
            ),
            "openrouter_vs_typesafe_total": len(parity_results) + 1 if ts_headers is not None else None,
        },
        "resolution_probes": res_results,
        "dimension_bound_probes": bound_results,
        "variant_probes": variant_results,
        "openrouter_inline_512": {
            "live_input_tokens": inline_512["input_tokens"],
            "short_key_a_input_tokens": 267 + tok.count_image(512, 512, "a"),
            "extra_tokens_from_uuid_and_array": inline_overhead_vs_short_key,
        },
        "billing_constant_probes": const_results,
        "parity_probes": parity_results,
        "box_version_and_endpoint_check": box_check,
    }

    out_path = ROOT / "evidence" / "image_and_parity_probes.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))

    assert report["summary"]["resolution_probes_matched"] == report["summary"]["resolution_probes_total"]
    assert report["summary"]["bound_probes_matched"] == report["summary"]["bound_probes_total"]
    assert report["summary"]["variant_probes_matched"] == report["summary"]["variant_probes_total"]
    assert report["summary"]["constant_probes_matched"] == report["summary"]["constant_probes_total"]
    assert report["summary"]["jev113_vs_jev114_matched"] == report["summary"]["jev113_vs_jev114_total"]
    assert not box_13["ok"] and box_13["status"] == 400 and box_14_or["ok"]
    if ts_headers is not None:
        assert report["summary"]["openrouter_vs_typesafe_matched"] == report["summary"]["openrouter_vs_typesafe_total"]
    print("ALL IMAGE, CONSTANT, VERSION-PARITY, AND ENDPOINT-PARITY PROBES PASSED 100%.")


if __name__ == "__main__":
    main()
