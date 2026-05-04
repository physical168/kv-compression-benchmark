import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
FINCH_SUM = Path("finch_4x4_summary.csv")
EA_KVZIP_SUM = Path("ea_kvzip_summary.csv")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

print("Loading", FINCH_SUM.name)
df_f = pd.read_csv(FINCH_SUM)

rows = []
for _, r in df_f.iterrows():
    method = "finch" if r['finch_enabled'] else "baseline"
    rows.append({
        "method": method,
        "use_cpt": bool(r["use_cpt"]),
        "compression_ratio": float(r["compression_ratio"]),
        "accuracy": float(r["accuracy"]),
        "latency_ms_mean": float(r["latency_ms_mean"]),
        "n": int(r["n"])
    })

print("Loading", EA_KVZIP_SUM.name)
df_ea = pd.read_csv(EA_KVZIP_SUM)
for _, r in df_ea.iterrows():
    rows.append({
        "method": r["method"],
        "use_cpt": bool(r["use_cpt"]),
        "compression_ratio": float(r["compression_ratio"]),
        "accuracy": float(r["accuracy"]),
        "latency_ms_mean": float(r["latency_ms_mean"]),
        "n": int(r["n"])
    })

df_all = pd.DataFrame(rows)

# For baseline, average across the "fake" ratios since it doesn't actually compress
baseline = df_all[df_all['method'] == 'baseline'].groupby('use_cpt').mean(numeric_only=True).reset_index()

# Methods to plot lines for
comp_methods = ["finch", "ea", "kvzip"]
colors = {"finch": "#2ca02c", "ea": "#ff7f0e", "kvzip": "#d62728"}
labels = {"finch": "FinchPress", "ea": "ExpectedAttention", "kvzip": "KVzipPress"}

def plot_metric(metric, ylabel, title, out_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    for i, use_cpt in enumerate([False, True]):
        ax = axes[i]
        
        # Plot Baseline as horizontal line
        bs_val = baseline[baseline['use_cpt'] == use_cpt][metric].iloc[0]
        ax.axhline(bs_val, color='blue', linestyle='--', linewidth=2, label=f"Baseline ({'CPT' if use_cpt else 'Full'})")
        
        # Plot each compression method
        for m in comp_methods:
            sub = df_all[(df_all['method'] == m) & (df_all['use_cpt'] == use_cpt)].sort_values('compression_ratio')
            if len(sub) == 0: continue
            ax.plot(sub['compression_ratio'], sub[metric], marker='o', linewidth=2, 
                    color=colors[m], label=labels[m])
            
        cpt_title = "With CPT (Context Truncated)" if use_cpt else "Without CPT (Full Text)"
        ax.set_title(cpt_title)
        ax.set_xlabel("Compression Ratio")
        if i == 0: ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    fig.suptitle(title, fontsize=14, y=1.05)
    fig.tight_layout()
    fp = OUT_DIR / out_name
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print("Saved", fp)
    plt.close()

plot_metric("accuracy", "Accuracy", "Qwen2.5-0.5B: Accuracy vs Compression Ratio (reviews_1000)", "benchmark_accuracy.png")
plot_metric("latency_ms_mean", "Latency (ms/sample)", "Qwen2.5-0.5B: Latency vs Compression Ratio (reviews_1000)", "benchmark_latency.png")

print("All charts generated and saved to figures/ folder.")
