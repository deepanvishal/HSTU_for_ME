# ============================================================
# PROV-VIS-05: Diagnosis Density on Trigger Day
# Distribution of how many distinct diagnoses appear on the
# same day as a qualified trigger
# Requires: dx_density DataFrame from Block 8b
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

GREEN    = "#059669"
TEXT_DARK = "#1F2937"
TEXT_MED  = "#6B7280"
GRID_CLR  = "#E5E7EB"
OUT = "./presentation_visuals/"

df = dx_density.copy()
df["n_dx_capped"] = df["n_dx"].clip(upper=20)
df_agg = df.groupby("n_dx_capped")["n_trigger_days"].sum().reset_index()
df_agg.loc[df_agg["n_dx_capped"] == 20, "n_dx_capped"] = 20

total = df_agg["n_trigger_days"].sum()
df_agg["pct"] = df_agg["n_trigger_days"] / total * 100

labels = [str(int(x)) if x < 20 else "20+" for x in df_agg["n_dx_capped"]]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(df_agg)), df_agg["pct"], color=GREEN, edgecolor="white", width=0.7)

for bar, pct, count in zip(bars, df_agg["pct"], df_agg["n_trigger_days"]):
    if pct > 2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{pct:.1f}%", ha="center", fontsize=7, color=TEXT_DARK)

ax.set_xticks(range(len(df_agg)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlabel("Distinct Diagnoses on Trigger Day")
ax.set_ylabel("% of Trigger-Days")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="y", color=GRID_CLR, linewidth=0.5)

med = int(stats['median_dx'])
ax.axvline(med - 1, color="#DC2626", linestyle="--", linewidth=1)
ax.text(med - 0.5, ax.get_ylim()[1] * 0.9, f"Median: {med}",
        fontsize=9, color="#DC2626", style="italic")

ax.set_title("How Many Diagnoses Accompany a Qualified Trigger?\n"
             "Distribution of distinct diagnoses billed on the trigger day "
             f"(n={total:,} trigger-days)")
plt.tight_layout()
plt.savefig(f"{OUT}prov_vis_05_dx_density.png")
plt.show()

print("Dx density visual done.")
