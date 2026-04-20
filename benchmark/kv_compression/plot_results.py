"""Generate figures for SnapKV benchmark report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── Data from experiments ─────────────────────────────────────────────────────

# Exp1: SnapKV algorithm breakdown (ratio=0.5)
exp1_sl = [4096, 8192, 16384, 32768, 65536]
exp1_ops = {
    "gqa_expand":  [0.076, 0.129, 0.234, 0.487, 1.151],
    "qk_matmul":   [0.079, 0.119, 0.209, 0.401, 0.770],
    "topk":        [0.150, 0.188, 0.469, 0.426, 0.454],
    "pool_sum":    [0.070, 0.070, 0.086, 0.129, 0.208],
    "transpose":   [0.048, 0.056, 0.077, 0.119, 0.203],
    "other":       [0.111, 0.104, 0.095, 0.098, 0.097],
}

# Exp2: Pipeline breakdown (ratio=0.5, pfx=80%)
exp2_sl = [4096, 8192, 16384, 32768]
exp2_phases = {
    "CPU transfer": [0.53, 0.91, 2.03, 5.08],
    "Pool IO":      [0.17, 0.19, 0.29, 0.51],
    "FA":           [0.29, 0.62, 1.49, 3.91],
    "Compress":     [0.46, 0.60, 1.08, 1.47],
    "Write":        [0.25, 0.22, 0.20, 0.31],
}

# Exp2: High prefix (pfx>=80%) phase proportions at 32K
exp2_high_pfx_data = {
    # (ratio, pfx): {phase: ms}
    (0.3, 0.8): {"xfer":4.82,"pool":0.49,"FA":3.92,"comp":1.49,"write":0.37},
    (0.3, 0.9): {"xfer":5.24,"pool":0.49,"FA":1.77,"comp":1.60,"write":0.37},
    (0.3, 1.0): {"xfer":6.07,"pool":0.49,"FA":0.92,"comp":1.46,"write":0.37},
    (0.5, 0.8): {"xfer":5.08,"pool":0.51,"FA":3.91,"comp":1.47,"write":0.31},
    (0.5, 0.9): {"xfer":5.68,"pool":0.51,"FA":1.76,"comp":1.54,"write":0.32},
    (0.5, 1.0): {"xfer":6.52,"pool":0.51,"FA":0.92,"comp":1.45,"write":0.31},
    (0.7, 0.8): {"xfer":4.87,"pool":0.49,"FA":3.92,"comp":1.44,"write":0.27},
    (0.7, 0.9): {"xfer":5.25,"pool":0.48,"FA":1.76,"comp":1.42,"write":0.26},
    (0.7, 1.0): {"xfer":5.86,"pool":0.54,"FA":1.12,"comp":1.62,"write":0.24},
}

# Exp3: CPU prefix vs recompute (ratio=0.5)
exp3_sl = [4096, 8192, 16384, 32768]
exp3_A = [1.72, 2.53, 4.74, 11.06]  # pfx=80%
exp3_B = [2.54, 5.57, 12.95, 43.32]

# Exp4: Memory
exp4_sl = [4096, 8192, 16384, 32768, 65536]
exp4_A = [134.9, 240.3, 439.0, 836.7, 1632.0]
exp4_B = [203.7, 366.1, 692.8, 1344.8, 2649.4]

colors = plt.cm.Set2.colors

# ── Figure 1: SnapKV algorithm stacked bar ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(exp1_sl))
w = 0.5
bottom = np.zeros(len(exp1_sl))
for i, (op, vals) in enumerate(exp1_ops.items()):
    ax.bar(x, vals, w, bottom=bottom, label=op, color=colors[i % len(colors)])
    bottom += np.array(vals)
ax.set_xticks(x); ax.set_xticklabels([f"{s//1024}K" for s in exp1_sl])
ax.set_xlabel("Sequence Length"); ax.set_ylabel("Time (ms)")
ax.set_title("SnapKV Algorithm Sub-operation Breakdown (ratio=0.5, no-softmax)")
ax.legend(loc="upper left"); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_snapkv_breakdown.png", dpi=150)
print(f"Saved {OUT}/fig1_snapkv_breakdown.png")

# ── Figure 2: Pipeline stacked bar (pfx=80%, ratio=0.5) ──────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(exp2_sl)); w = 0.5
bottom = np.zeros(len(exp2_sl))
for i, (ph, vals) in enumerate(exp2_phases.items()):
    ax.bar(x, vals, w, bottom=bottom, label=ph, color=colors[i % len(colors)])
    bottom += np.array(vals)
ax.set_xticks(x); ax.set_xticklabels([f"{s//1024}K" for s in exp2_sl])
ax.set_xlabel("Sequence Length"); ax.set_ylabel("Time (ms)")
ax.set_title("Pipeline Phase Breakdown (ratio=0.5, pfx=80%)")
ax.legend(loc="upper left"); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_pipeline_breakdown.png", dpi=150)
print(f"Saved {OUT}/fig2_pipeline_breakdown.png")

# ── Figure 3: High prefix phase proportions (32K, stacked %) ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for idx, ratio in enumerate([0.3, 0.5, 0.7]):
    ax = axes[idx]
    pfxs = [0.8, 0.9, 1.0]
    phase_names = ["xfer", "pool", "FA", "comp", "write"]
    phase_labels = ["CPU Transfer", "Pool IO", "FlashAttn", "Compress", "Write"]
    x = np.arange(len(pfxs))
    bottom = np.zeros(len(pfxs))
    for pi, pn in enumerate(phase_names):
        vals = [exp2_high_pfx_data[(ratio, p)][pn] for p in pfxs]
        ax.bar(x, vals, 0.5, bottom=bottom, label=phase_labels[pi] if idx==0 else "", color=colors[pi])
        bottom += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels([f"{int(p*100)}%" for p in pfxs])
    ax.set_xlabel("CPU Prefix Match Rate")
    ax.set_title(f"ratio={ratio}")
    if idx == 0: ax.set_ylabel("Time (ms)")
axes[0].legend(loc="upper right", fontsize=8)
fig.suptitle("32K Pipeline: Phase Distribution at High Prefix Match Rates", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_high_prefix_phases.png", dpi=150)
print(f"Saved {OUT}/fig3_high_prefix_phases.png")

# ── Figure 4: CPU prefix vs recompute ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(exp3_sl)); w = 0.3
ax1.bar(x - w/2, exp3_A, w, label="CPU Prefix (pfx=80%)", color=colors[0])
ax1.bar(x + w/2, exp3_B, w, label="Full Recompute", color=colors[1])
ax1.set_xticks(x); ax1.set_xticklabels([f"{s//1024}K" for s in exp3_sl])
ax1.set_xlabel("Sequence Length"); ax1.set_ylabel("Time (ms)")
ax1.set_title("Latency: CPU Prefix vs Full Recompute")
ax1.legend(); ax1.grid(axis='y', alpha=0.3)

x2 = np.arange(len(exp4_sl)); w2 = 0.3
ax2.bar(x2 - w2/2, [m/1024 for m in exp4_A], w2, label="CPU Prefix", color=colors[0])
ax2.bar(x2 + w2/2, [m/1024 for m in exp4_B], w2, label="Full Recompute", color=colors[1])
ax2.set_xticks(x2); ax2.set_xticklabels([f"{s//1024}K" for s in exp4_sl])
ax2.set_xlabel("Sequence Length"); ax2.set_ylabel("Peak Memory (GB)")
ax2.set_title("Peak GPU Memory Comparison")
ax2.legend(); ax2.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_prefix_vs_recompute.png", dpi=150)
print(f"Saved {OUT}/fig4_prefix_vs_recompute.png")

# ── Figure 5: Phase proportion pie at 32K pfx=80% ratio=0.5 ──────────────────
fig, ax = plt.subplots(figsize=(7, 7))
d = exp2_high_pfx_data[(0.5, 0.8)]
labels = ["CPU Transfer", "Pool IO", "FlashAttn", "Compress", "Write"]
sizes = [d["xfer"], d["pool"], d["FA"], d["comp"], d["write"]]
total = sum(sizes)
pcts = [s/total*100 for s in sizes]
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
    colors=colors[:5], startangle=90, textprops={'fontsize': 11})
ax.set_title(f"Pipeline Phase Distribution\n(32K, ratio=0.5, pfx=80%, total={total:.1f}ms)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_pie_32k.png", dpi=150)
print(f"Saved {OUT}/fig5_pie_32k.png")

# ── Figure 6: Transfer & Compress proportion vs prefix ratio ──────────────────
fig, ax = plt.subplots(figsize=(10, 5))
pfxs = [0.8, 0.9, 1.0]
for ratio in [0.3, 0.5, 0.7]:
    xfer_pcts = []
    comp_pcts = []
    for p in pfxs:
        d = exp2_high_pfx_data[(ratio, p)]
        t = sum(d.values())
        xfer_pcts.append(d["xfer"]/t*100)
        comp_pcts.append(d["comp"]/t*100)
    ax.plot(pfxs, xfer_pcts, 'o-', label=f"Transfer (r={ratio})", linewidth=2)
    ax.plot(pfxs, comp_pcts, 's--', label=f"Compress (r={ratio})", linewidth=2)
ax.set_xlabel("CPU Prefix Match Rate"); ax.set_ylabel("Proportion (%)")
ax.set_title("Transfer & Compress Proportion at High Prefix Match (32K)")
ax.set_xticks(pfxs); ax.set_xticklabels([f"{int(p*100)}%" for p in pfxs])
ax.legend(ncol=2, fontsize=9); ax.grid(alpha=0.3)
ax.set_ylim(0, 80)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_xfer_compress_proportion.png", dpi=150)
print(f"Saved {OUT}/fig6_xfer_compress_proportion.png")

print("\nAll figures saved to", OUT)
