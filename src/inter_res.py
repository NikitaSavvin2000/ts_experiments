"""
pdm run src/inter_res.py
"""

from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_and_merge(results_path):
    base = Path(results_path)
    files = [p for p in base.rglob("line_result_table.csv") if base in p.parents]
    dfs = [pd.read_csv(f) for f in files]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


results_path_init = "/Users/nikitasavvin/Desktop/PhD/ts_experiments/export/test_optuna_v2"
results_path = f"{results_path_init}/results"
article_materials_path = os.path.join(results_path_init, "article_materials")
os.makedirs(article_materials_path, exist_ok=True)

df = load_and_merge(results_path)

cols_to_select = ["type", "dataset_name", "trajectory_cols", "r2" , "mape", "rmse" ,"mae", "model"]


df = df[cols_to_select]

# trajectory = [
#     "baseline",
#     "calendar_components",
#     "engineered_datetime_features",
#     "stat_selected_features",
#     "horizon_selected_features"
# ]

# trajectory = ["baseline", "calendar_components",  "engineered_datetime_features", "stat_selected_features", "optuna", "сlassic_GA", "GA_horizon_selected_features"]
trajectory = ["baseline", "calendar_components", "engineered_datetime_features", "mi_features", "chi_features", "pearson_features", "сlassic_GA", "optuna_features", "GA_horizon_selected_features",]


order_list = df["trajectory_cols"].unique().tolist()

order = [x for x in trajectory if x in order_list] + [x for x in order_list if x not in trajectory]

df["trajectory_cols"] = pd.Categorical(df["trajectory_cols"], categories=order, ordered=True)

df.to_csv(f"{article_materials_path}/all_metrix.csv")


df_long = df.melt(
    id_vars=["type", "dataset_name", "trajectory_cols", "model"],
    value_vars=["mape", "r2"],
    var_name="metric",
    value_name="value"
)

out = (
    df_long.pivot_table(
        index=["type", "dataset_name", "trajectory_cols", "metric"],
        columns="model",
        values="value",
        aggfunc="first"
    )
    .reset_index()
    .rename(columns={
        "dataset_name": "dataset",
        "trajectory_cols": "trajectory"
    })
    .sort_values(["type", "dataset", "trajectory", "metric"])
)

out.columns.name = None


mape = out[out["metric"] == "mape"].copy()
r2 = out[out["metric"] == "r2"].copy()

model_cols = [c for c in mape.columns if c not in ["type", "dataset", "trajectory", "metric"]]

tmp = mape.melt(
    id_vars=["type", "dataset", "trajectory"],
    value_vars=model_cols,
    var_name="model",
    value_name="mape"
)

best = tmp.loc[tmp.groupby(["type", "dataset"])["mape"].idxmin()].copy()

r2_tmp = r2.melt(
    id_vars=["type", "dataset", "trajectory"],
    value_vars=model_cols,
    var_name="model",
    value_name="r2"
)

best = best.merge(
    r2_tmp,
    on=["type", "dataset", "trajectory", "model"],
    how="left"
)[["type", "dataset", "trajectory", "model", "mape", "r2"]].reset_index(drop=True)


# print(out)
# print("="*100)
# print(best)


out.to_csv(f"{article_materials_path}/all_metrix_table.csv")

best.to_csv(f"{article_materials_path}/best_table.csv")






all_types = best["type"].unique()

for current_type in all_types:
    fig = plt.figure(figsize=(6 * (len(best["dataset"].unique()) + 1), 3 * len(order)))


    df_datasets = best[best["type"] == current_type].reset_index(drop=True)

    n_rows = len(order)
    n_cols = len(df_datasets) + 1

    gs = GridSpec(
        n_rows,
        n_cols,
        width_ratios=[0.01] + [1] * len(df_datasets),
        hspace=0.25,
        wspace=0.25
    )

    row_global = 0

    for trajectory in order:

        ax_label = fig.add_subplot(gs[row_global, 0])
        ax_label.axis("off")
        ax_label.text(
            0.5,
            0.5,
            trajectory,
            fontsize=11,
            rotation=90,
            ha="center",
            va="center"
        )

        for col_idx, (_, row) in enumerate(df_datasets.iterrows(), start=1):

            dataset = row["dataset"]
            model = row["model"]

            path = f"{results_path}/{dataset}_{model}_{trajectory}/test_pred_norm.csv"
            path_metrics = f"{results_path}/{dataset}_{model}_{trajectory}/line_result_table.csv"

            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            df["datetime"] = pd.to_datetime(df["datetime"])

            df_metrics = pd.read_csv(path_metrics)
            mape = float(df_metrics["mape"].iloc[0])
            r2 = float(df_metrics["r2"].iloc[0])

            ax = fig.add_subplot(gs[row_global, col_idx])

            ax.plot(df["datetime"], df["true"], label="true")
            ax.plot(df["datetime"], df["pred"], label="pred")

            ax.set_xlabel("datetime")
            ax.set_ylabel("value")
            ax.set_xticks([])

            is_first_row = (row_global == 0)

            if is_first_row:
                ax.set_title(f"{dataset}\n{model}", pad=18)
                ax.text(
                    0.5,
                    1.02,
                    f"MAPE: {mape:.3f} | R2: {r2:.3f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=10
                )
            else:
                ax.set_title("")
                ax.text(
                    0.5,
                    1.05,
                    f"MAPE: {mape:.3f} | R2: {r2:.3f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=10
                )

        row_global += 1

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    out_path = os.path.join(article_materials_path, f"{current_type}.png")

    fig.savefig(
        out_path,
        dpi=400,
        bbox_inches="tight",
        format="png"
    )