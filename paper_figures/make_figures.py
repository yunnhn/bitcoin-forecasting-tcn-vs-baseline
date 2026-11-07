"""Gambar grafik paper dari figdata.npz: huruf Times, panel sejajar, skala seragam.

Menulis PDF untuk LaTeX dan PNG 600 dpi untuk Word. Butuh font Times New Roman.

    python make_figures.py [--pdf-dir FOLDER] [--png-dir FOLDER]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
MODELS = ["RNN", "LSTM", "TCN"]
BLUE, ORANGE = "#1F5FBF", "#D9730D"
INK, MUTED, GRID = "#111111", "#555555", "#E3E3E3"
COL_W = 252.0 / 72  # lebar kolom IEEEtran, dalam inci

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "pdf.fonttype": 42, "legend.frameon": False,
})


def idn(value, dec=0):
    """Format angka gaya Indonesia: titik ribuan, koma desimal."""
    return f"{value:,.{dec}f}".replace(",", "#").replace(".", ",").replace("#", ".").replace("-", "−")


def panel_title(ax, text):
    ax.set_title(text, loc="left")


def fig2(d):
    dates = pd.date_range("2023-09-18", periods=len(d["y"]), freq="D")
    assert dates[-1] == pd.Timestamp("2025-12-03")
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 4.75), sharex=True, sharey=True, layout="constrained")
    for ax, name, tag in zip(axes, MODELS, "abc"):
        ax.plot(dates, d["y"] / 1000, color=MUTED, lw=0.9, alpha=0.75)
        ax.plot(dates, d[f"pred_{name}"] / 1000, color=BLUE, lw=0.9)
        panel_title(ax, f"({tag}) {name}: RMSE {idn(float(d[f'best_rmse_{name}']), 2)} USD")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(labelbottom=True)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Harga (ribu USD)")
    axes[0].set_xlim(dates[0], dates[-1])
    fig.legend(handles=[Line2D([], [], color=MUTED, lw=1.2, label="Aktual"),
                        Line2D([], [], color=BLUE, lw=1.2, label="Prediksi")],
               loc="outside upper center", ncol=2)
    return fig


def fig3(d):
    label = {"Close Price": "Harga penutupan", "Year_Sin": "year_sin", "Year_Cos": "year_cos",
             "Month_Sin": "month_sin", "Month_Cos": "month_cos"}
    shap, x = d["shap_usd"], d["x_aggr"]
    mean = np.abs(shap).mean(axis=0)
    order = np.argsort(mean)[::-1]
    y = np.arange(len(order))
    names = [label[str(d["features"][i])] for i in order]
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.1), layout="constrained")

    a.barh(y, mean[order], height=0.62, color=BLUE)
    for row, value in zip(y, mean[order]):
        a.text(value + 800, row, idn(float(value)), va="center", fontsize=7.5)
    panel_title(a, "(a) Rata-rata besaran kontribusi")
    a.set_xlim(0, 65000)
    a.set_xlabel("Rata-rata |nilai SHAP| (USD)")

    cmap = LinearSegmentedColormap.from_list("nilai", ["#BFD3F2", BLUE, "#0B2A5B"])
    span = np.ptp(x, axis=0)
    norm = (x - x.min(axis=0)) / np.where(span == 0, 1, span)
    b.axvline(0, color=MUTED, lw=0.6, zorder=1)
    points = None
    for row, i in zip(y, order):
        points = b.scatter(shap[:, i], np.full(len(shap), row), c=norm[:, i], cmap=cmap, vmin=0, vmax=1,
                           s=20, edgecolors=INK, linewidths=0.35, zorder=3)
    panel_title(b, "(b) Nilai per sampel")
    b.set_xlim(-5000, 110000)
    b.set_xlabel("Nilai SHAP per sampel (USD)")
    # Bilah warna diletakkan di area kosong agar lebar kedua panel tetap sama
    holder = b.inset_axes([0.58, 0.26, 0.38, 0.07])
    bar = fig.colorbar(points, cax=holder, orientation="horizontal")
    bar.set_ticks([0, 1], labels=["Rendah", "Tinggi"])
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(length=0, labelsize=7)
    holder.set_title("Nilai fitur", fontsize=7.5, pad=2)

    for ax in (a, b):
        ax.set_yticks(y, names)
        ax.set_ylim(len(y) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: idn(v)))
    return fig


def fig4(d):
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 4.4), sharex=True, sharey=True, layout="constrained")
    naik = d["labels"] == "Naik (Uptrend)"
    for ax, name, tag in zip(axes, MODELS, "abc"):
        t = d[f"tsne_{name}"]
        # Urutan gambar diacak agar tidak ada kelas yang selalu menutupi kelas lain
        for i in np.random.default_rng(42).permutation(len(t)):
            ax.scatter(t[i, 0], t[i, 1], s=6 if naik[i] else 7, marker="o" if naik[i] else "^",
                       color=BLUE if naik[i] else ORANGE, alpha=0.75, linewidths=0)
        panel_title(ax, f"({tag}) {name}")
        ax.tick_params(labelbottom=True)
        ax.set_ylabel("t-SNE 2")
    axes[0].set_xlim(-55, 55)
    axes[0].set_ylim(-42, 42)
    axes[-1].set_xlabel("t-SNE 1")
    fig.legend(handles=[
        Line2D([], [], ls="", marker="o", color=BLUE, ms=4, label=f"Naik ({int(naik.sum())} hari)"),
        Line2D([], [], ls="", marker="^", color=ORANGE, ms=4, label=f"Turun ({int((~naik).sum())} hari)"),
    ], loc="outside upper center", ncol=2)
    return fig


def fig5(d):
    t = np.arange(1, 61)
    maps = d["maps"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 2.9), sharex=True, layout="constrained",
                               height_ratios=[1, 1.15])
    a.plot(t, d["x_last"], color=INK, lw=1)
    panel_title(a, "(a) Input: harga penutupan pada skala normalisasi")
    a.set_ylabel("Nilai ternormalisasi")
    a.tick_params(labelbottom=True)
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: idn(v, 1)))

    b.plot(t, maps.mean(axis=-1), color=BLUE, lw=1.3, label="Rata-rata 32 filter")
    for i, style in enumerate(["--", ":", "-."]):
        b.plot(t, maps[:, i], color=MUTED, lw=0.8, ls=style, label=f"Filter {i}")
    panel_title(b, "(b) Aktivasi lapisan konvolusi pertama TCN")
    b.set_xlabel("Timestep (hari ke-1 sampai ke-60)")
    b.set_ylabel("Aktivasi")
    b.set_xlim(1, 60)
    b.set_xticks([1, 10, 20, 30, 40, 50, 60])
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: idn(v, 1)))
    b.legend(loc="lower left", ncol=2, fontsize=7, bbox_to_anchor=(0.0, -0.02), handlelength=2.2,
             columnspacing=1.0, borderaxespad=0.2)
    b.set_ylim(-0.85, 0.42)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=HERE.parent / "paper_ieee_latex" / "figures")
    parser.add_argument("--png-dir", type=Path, default=HERE / "png")
    args = parser.parse_args()
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    args.png_dir.mkdir(parents=True, exist_ok=True)

    with np.load(HERE / "figdata.npz") as data:
        d = {key: data[key] for key in data.files}

    for name, build in [("fig2_prediksi", fig2), ("fig3_shap", fig3), ("fig4_tsne", fig4), ("fig5_aktivasi", fig5)]:
        fig = build(d)
        fig.savefig(args.pdf_dir / f"{name}.pdf")
        fig.savefig(args.png_dir / f"{name}.png", dpi=600)
        plt.close(fig)
        print("tersimpan", name)


if __name__ == "__main__":
    main()
