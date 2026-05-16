import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

np.random.seed(42)

# Simulates M/M/1 queue: arrival rate λ=0.8, service rate μ=1.0
def simulate_mm1_queue(n_customers=2000, lam=0.8, mu=1.0):
    """Simulate M/M/1 queue and return waiting times."""
    inter_arrivals = np.random.exponential(1/lam, n_customers)
    service_times  = np.random.exponential(1/mu,  n_customers)
    arrival_times  = np.cumsum(inter_arrivals)

    wait_times = np.zeros(n_customers)
    departure   = 0.0
    for i in range(n_customers):
        start       = max(arrival_times[i], departure)
        wait_times[i] = start - arrival_times[i]
        departure   = start + service_times[i]

    return wait_times

# One long run (for Welch & Batch Means)
long_run = simulate_mm1_queue(n_customers=2000)

# Multiple replications (for Replication-Deletion)
N_REPS   = 20
REP_LEN  = 500
replications = [simulate_mm1_queue(n_customers=REP_LEN) for _ in range(N_REPS)]

print("=" * 60)
print("  PT: Output Analysis — Three Methods")
print("=" * 60)

# METHOD 1: WELCH METHOD
print("\n[1] WELCH METHOD")

WELCH_REPS = 10
WELCH_LEN  = 500
welch_runs = np.array([simulate_mm1_queue(n_customers=WELCH_LEN)
                        for _ in range(WELCH_REPS)])

# Step 1 – average across replications at each time point
Y_bar = welch_runs.mean(axis=0)           # shape (WELCH_LEN,)

# Step 2 – moving average with window w
def moving_average(data, w):
    n = len(data)
    ma = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - w)
        hi = min(n - 1, i + w)
        ma[i] = data[lo:hi+1].mean()
    return ma

w = 20
MA = moving_average(Y_bar, w)

# Step 3 – detect warm-up: first index where MA stabilises (within 5% of tail mean)
tail_mean   = MA[WELCH_LEN // 2:].mean()
stable_mask = np.abs(MA - tail_mean) / tail_mean < 0.05
warmup_idx  = int(np.argmax(stable_mask)) if stable_mask.any() else WELCH_LEN // 4

# Step 4 – steady-state estimate (after warm-up)
steady_data      = long_run[warmup_idx:]
welch_mean       = steady_data.mean()
welch_ci         = stats.t.interval(0.95, df=len(steady_data)-1,
                                    loc=welch_mean,
                                    scale=stats.sem(steady_data))

print(f"  Detected warm-up period  : {warmup_idx} observations")
print(f"  Steady-state obs used    : {len(steady_data)}")
print(f"  Mean waiting time        : {welch_mean:.4f}")
print(f"  95% CI                   : ({welch_ci[0]:.4f}, {welch_ci[1]:.4f})")

# METHOD 2 ── REPLICATION-DELETION APPROACH
# Purpose : Use multiple independent replications; delete warm-up

print("\n[2] REPLICATION-DELETION APPROACH")

DELETE_FRACTION = 0.20          # delete first 20 % of each replication
delete_n        = int(REP_LEN * DELETE_FRACTION)

rep_means = []
for i, rep in enumerate(replications):
    trimmed = rep[delete_n:]    # delete warm-up
    rep_means.append(trimmed.mean())

rep_means   = np.array(rep_means)
rd_mean     = rep_means.mean()
rd_ci       = stats.t.interval(0.95, df=N_REPS - 1,
                               loc=rd_mean,
                               scale=stats.sem(rep_means))

print(f"  Replications             : {N_REPS}")
print(f"  Obs deleted per rep      : {delete_n}  ({int(DELETE_FRACTION*100)}%)")
print(f"  Obs kept per rep         : {REP_LEN - delete_n}")
print(f"  Grand mean               : {rd_mean:.4f}")
print(f"  95% CI                   : ({rd_ci[0]:.4f}, {rd_ci[1]:.4f})")

# Per-replication summary table
rd_df = pd.DataFrame({
    "Replication": range(1, N_REPS + 1),
    "Mean (after deletion)": np.round(rep_means, 4)
})
print(f"\n{rd_df.to_string(index=False)}")

# METHOD 3 ── BATCH MEANS METHOD
# Purpose : Divide one long run (after warm-up) into k non-overlapping
print("\n[3] BATCH MEANS METHOD")

K          = 30          # number of batches
bm_data    = long_run[warmup_idx:]          # post-warm-up data
batch_size = len(bm_data) // K
trimmed    = bm_data[:batch_size * K]       # discard leftover

batches     = trimmed.reshape(K, batch_size)
batch_means = batches.mean(axis=1)

bm_mean = batch_means.mean()
bm_ci   = stats.t.interval(0.95, df=K - 1,
                            loc=bm_mean,
                            scale=stats.sem(batch_means))

print(f"  Post-warmup observations : {len(bm_data)}")
print(f"  Number of batches (k)    : {K}")
print(f"  Batch size               : {batch_size}")
print(f"  Mean of batch means      : {bm_mean:.4f}")
print(f"  95% CI                   : ({bm_ci[0]:.4f}, {bm_ci[1]:.4f})")
print(f"  Std dev of batch means   : {batch_means.std():.4f}")

# COMPARISON TABLE
print("\n" + "=" * 60)
print("  COMPARISON SUMMARY")
print("=" * 60)
theoretical = 1 / (1.0 - 0.8)   # E[W] for M/M/1 = λ/μ(μ-λ) → 4.0
comp_df = pd.DataFrame({
    "Method":    ["Welch", "Replication-Deletion", "Batch Means", "Theoretical"],
    "Mean":      [round(welch_mean,4), round(rd_mean,4), round(bm_mean,4), round(theoretical,4)],
    "CI Lower":  [round(welch_ci[0],4), round(rd_ci[0],4), round(bm_ci[0],4), "—"],
    "CI Upper":  [round(welch_ci[1],4), round(rd_ci[1],4), round(bm_ci[1],4), "—"],
})
print(comp_df.to_string(index=False))

# PLOTS
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#0f172a")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

TEAL   = "#2dd4bf"
AMBER  = "#fbbf24"
ROSE   = "#f43f5e"
SLATE  = "#94a3b8"
BG     = "#1e293b"
WHITE  = "#f1f5f9"

def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=WHITE, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=SLATE)
    ax.xaxis.label.set_color(SLATE)
    ax.yaxis.label.set_color(SLATE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

# ── Plot 1: Welch moving average
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(Y_bar, color=SLATE, alpha=0.4, lw=1, label="Avg across reps")
ax1.plot(MA,    color=TEAL,  lw=2,        label=f"Moving avg (w={w})")
ax1.axvline(warmup_idx, color=ROSE, lw=2, ls="--", label=f"Warm-up end = {warmup_idx}")
ax1.axhline(tail_mean,  color=AMBER, lw=1, ls=":",  label=f"Steady mean = {tail_mean:.2f}")
style_ax(ax1, "Method 1 — Welch: Detecting Warm-Up")
ax1.set_xlabel("Customer index")
ax1.set_ylabel("Avg waiting time")
ax1.legend(fontsize=7, facecolor=BG, labelcolor=WHITE)

# ── Plot 2: Replication-Deletion per-rep means
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(range(1, N_REPS + 1), rep_means, color=TEAL, alpha=0.8, edgecolor=BG)
ax2.axhline(rd_mean,    color=AMBER, lw=2, ls="--", label=f"Grand mean = {rd_mean:.3f}")
ax2.axhline(rd_ci[0],  color=ROSE,  lw=1, ls=":",  label=f"95% CI")
ax2.axhline(rd_ci[1],  color=ROSE,  lw=1, ls=":")
ax2.fill_between(range(1, N_REPS + 2), rd_ci[0], rd_ci[1], color=ROSE, alpha=0.08)
style_ax(ax2, "Method 2 — Replication-Deletion: Per-Rep Means")
ax2.set_xlabel("Replication #")
ax2.set_ylabel("Mean waiting time")
ax2.legend(fontsize=8, facecolor=BG, labelcolor=WHITE)

# ── Plot 3: Batch Means
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(range(1, K + 1), batch_means, color=AMBER, alpha=0.8, edgecolor=BG)
ax3.axhline(bm_mean,   color=TEAL, lw=2, ls="--", label=f"Grand mean = {bm_mean:.3f}")
ax3.axhline(bm_ci[0],  color=ROSE, lw=1, ls=":",  label="95% CI")
ax3.axhline(bm_ci[1],  color=ROSE, lw=1, ls=":")
ax3.fill_between(range(1, K + 2), bm_ci[0], bm_ci[1], color=ROSE, alpha=0.08)
style_ax(ax3, "Method 3 — Batch Means: Batch-Level Means")
ax3.set_xlabel("Batch #")
ax3.set_ylabel("Batch mean waiting time")
ax3.legend(fontsize=8, facecolor=BG, labelcolor=WHITE)

# ── Plot 4: Comparison of CIs
ax4 = fig.add_subplot(gs[1, 1])
methods = ["Welch", "Rep-Deletion", "Batch Means"]
means   = [welch_mean, rd_mean, bm_mean]
lowers  = [welch_ci[0], rd_ci[0], bm_ci[0]]
uppers  = [welch_ci[1], rd_ci[1], bm_ci[1]]
colors  = [TEAL, AMBER, ROSE]
y_pos   = [2, 1, 0]

for idx, (m, lo, hi, _up) in enumerate(zip(methods, means, lowers, uppers)):
    c = colors[idx]
    y = y_pos[idx]
    ax4.plot([lo, hi], [y, y], lw=4, color=c, solid_capstyle="round")
    ax4.scatter(m, y, zorder=5, s=80, color=c, edgecolors=WHITE, lw=1.5)

ax4.axvline(theoretical, color=SLATE, lw=1.5, ls="--",
            label=f"Theoretical E[W] = {theoretical:.2f}")
ax4.set_yticks(y_pos)
ax4.set_yticklabels(methods, color=WHITE)
style_ax(ax4, "Comparison of 95% Confidence Intervals")
ax4.set_xlabel("Mean waiting time")
ax4.legend(fontsize=8, facecolor=BG, labelcolor=WHITE)

fig.suptitle("Output Analysis — M/M/1 Queue Simulation\n"
             "Welch Method  |  Replication-Deletion  |  Batch Means",
             color=WHITE, fontsize=13, fontweight="bold", y=0.98)
import os

# Create an 'outputs' folder in your current directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Save using a clean, cross-platform path
plt.savefig("outputs/output_analysis_plots.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            
plt.close()
print("\n✓ Plot saved → output_analysis_plots.png")
