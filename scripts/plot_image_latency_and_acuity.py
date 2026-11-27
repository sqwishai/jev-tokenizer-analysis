from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    data = json.loads((ROOT / "evidence" / "image_latency_and_acuity.json").read_text(encoding="utf-8"))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )

    jp = data["jev_prose_ms"]
    qp = data["qwen_prose_ttft_ms"]
    jm_min, jb_min = np.polyfit([r["billed_tokens"] for r in jp], [r["min_ms"] for r in jp], 1)
    jm_med, jb_med = np.polyfit([r["billed_tokens"] for r in jp], [r["med_ms"] for r in jp], 1)
    qm_min, qb_min = np.polyfit([r["prompt_tokens"] for r in qp], [r["min_ms"] for r in qp], 1)
    qm_med, qb_med = np.polyfit([r["prompt_tokens"] for r in qp], [r["med_ms"] for r in qp], 1)

    four_imgs = [r for r in data["image_latency_comparison"] if r["count"] == 4]
    widths = np.array([r["width"] for r in four_imgs])
    q_vis = np.array([r["qwen_native_tokens"] for r in four_imgs])
    j_bill = np.array([r["jev_billed_tokens"] for r in four_imgs])
    j_min = np.array([r["jev_min_ms"] for r in four_imgs])
    j_med = np.array([r["jev_med_ms"] for r in four_imgs])
    q_min = np.array([r["qwen_min_ms"] for r in four_imgs])
    q_med = np.array([r["qwen_med_ms"] for r in four_imgs])

    j_d_min = j_min - jb_min
    j_d_med = j_med - jb_med
    q_d_min = np.maximum(0.0, q_min - qb_min)
    q_d_med = np.maximum(0.0, q_med - qb_med)

    j_eq_min = j_d_min / jm_min
    j_eq_med = j_d_med / jm_med
    q_eq_min = q_d_min / qm_min
    q_eq_med = q_d_med / qm_med

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.8), dpi=180)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    x_k = q_vis / 1000.0
    ax.plot(x_k, x_k, color="#9ca3af", ls=":", lw=1.6, label="1.0x native Qwen 3.5 tokens (no ViT)")
    ax.plot(x_k, j_bill / 1000.0, color="#d97706", ls="--", lw=2.0, label="Jev 1.14 billed (~2.6x native)")
    ax.plot(x_k, j_eq_min / 1000.0, "o-", color="#dc2626", lw=2.3, ms=6.5, label="Jev 1.14 latency equiv. (min-med)")
    ax.fill_between(
        x_k,
        np.minimum(j_eq_min, j_eq_med) / 1000.0,
        np.maximum(j_eq_min, j_eq_med) / 1000.0,
        color="#dc2626",
        alpha=0.14,
    )
    ax.plot(
        x_k[2:],
        q_eq_min[2:] / 1000.0,
        "s-",
        color="#2563eb",
        lw=2.3,
        ms=6.5,
        label="Qwen3.5-35B-A3B TTFT equiv. (min-med)",
    )
    ax.fill_between(
        x_k[2:],
        np.minimum(q_eq_min[2:], q_eq_med[2:]) / 1000.0,
        np.maximum(q_eq_min[2:], q_eq_med[2:]) / 1000.0,
        color="#2563eb",
        alpha=0.14,
    )

    for idx_pt, lbl, offs in [(2, "4x768", (-24, 14)), (3, "4x1024", (-28, 16)), (4, "4x1280", (-32, 16)), (5, "4x1568", (-38, 10))]:
        ax.annotate(
            lbl,
            xy=(x_k[idx_pt], j_bill[idx_pt] / 1000.0),
            xytext=offs,
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#374151",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.8),
        )

    ax.set_title("A. Compute in Equivalent Prose Tokens\n(hardware speed canceled out)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Native Qwen 3.5 visual tokens (k, 4 images)")
    ax.set_ylabel("Equivalent prose tokens (k)")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 30.0)
    ax.legend(loc="upper left", fontsize=8.2, frameon=True, framealpha=0.95)

    ax2 = axes[1]
    slope_ratio = qm_min / jm_min
    l1, = ax2.plot(widths, j_d_min, "o-", color="#dc2626", lw=2.4, ms=6.5, label="Jev 1.14 compute +ms (left, min)")
    l2, = ax2.plot(
        widths[2:],
        q_d_min[2:] / slope_ratio,
        "s--",
        color="#2563eb",
        lw=2.2,
        ms=6.5,
        label=f"Qwen3.5-35B-A3B +ms / {slope_ratio:.2f} (right, min)",
    )
    w_fine = np.linspace(280, 1568, 100)
    l3, = ax2.plot(w_fine, j_d_min[-1] * (w_fine / 1568.0) ** 2, ":", color="#6b7280", lw=1.4, label="Pure quadratic (W^2) scaling")
    ax2.set_title("B. Image Compute Latency vs. Resolution\n(no downsampling cap up to 1568 px)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Image side length W (px, 4 x W x W PNGs)")
    ax2.set_ylabel("Jev 1.14 compute above 0-tok floor (ms)", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax2.set_xlim(220, 1650)
    ax2.set_ylim(-5, 155)
    ax2.set_xticks([280, 512, 768, 1024, 1280, 1568])

    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.set_ylim(-5 * slope_ratio, 155 * slope_ratio)
    ax2_r.set_ylabel(f"Qwen3.5-35B-A3B Parasail +ms ({slope_ratio:.2f}x scale)", color="#2563eb")
    ax2_r.tick_params(axis="y", labelcolor="#2563eb")
    ax2.legend(handles=[l1, l2, l3], loc="upper left", fontsize=8.2, frameon=True, framealpha=0.95)

    ax3 = axes[2]
    ac = data["optical_acuity_digits_4choice"]
    fpx = np.array(ac["font_px"])
    jev_acc = np.array(ac["jev_114_native_by_canvas"]["1568"]["acc_of_4"]) / 4.0 * 100.0
    q100_acc = np.array(ac["qwen35_35b_a3b_on_1568_by_scale"]["1.00"]["acc_of_4"]) / 4.0 * 100.0
    q075_acc = np.array(ac["qwen35_35b_a3b_on_1568_by_scale"]["0.75"]["acc_of_4"]) / 4.0 * 100.0
    q050_acc = np.array(ac["qwen35_35b_a3b_on_1568_by_scale"]["0.50"]["acc_of_4"]) / 4.0 * 100.0

    ax3.axhline(25, color="#9ca3af", ls=":", lw=1.3, label="Random chance (25%)")
    ax3.plot(fpx, jev_acc, "o-", color="#dc2626", lw=2.6, ms=7.5, label="Jev 1.14 native 1568x1568")
    ax3.plot(fpx, q100_acc, "s--", color="#2563eb", lw=2.2, ms=6.5, label="Qwen3.5-35B-A3B @ 1.00x (2455 tok)")
    ax3.plot(fpx, q075_acc, "^-", color="#059669", lw=2.0, ms=6.5, label="Qwen3.5-35B-A3B @ 0.75x (1423 tok)")
    ax3.plot(fpx, q050_acc, "v-", color="#7c3aed", lw=2.0, ms=6.5, label="Qwen3.5-35B-A3B @ 0.50x (630 tok)")
    ax3.set_title("C. Optical Acuity on 1568x1568\n(4-choice 2-digit font test)", fontsize=11, fontweight="bold", pad=10)
    ax3.set_xlabel("Rendered font size on 1568x1568 canvas (px)")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_xticks(fpx)
    ax3.set_xticklabels([f"{f}px" for f in fpx])
    ax3.set_ylim(15, 110)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=8.0, frameon=True, framealpha=0.95)

    fig.suptitle(
        "Jev 1.14 vs. Qwen3.5-35B-A3B: Image Prefill Latency & Resolution Acuity (1.00x Native, No Downsampling)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.28, wspace=0.33)
    out = ROOT / "evidence" / "image_latency_and_acuity.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
