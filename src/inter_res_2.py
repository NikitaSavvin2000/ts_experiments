"""
pdm run src/inter_res_2.py
"""

from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go


import os
import pandas as pd
import matplotlib.pyplot as plt

# trajectory = ["baseline", "calendar_components",  "engineered_datetime_features", "stat_selected_features", "optuna", "сlassic_GA", "GA_horizon_selected_features"]
trajectory = ["baseline", "engineered_datetime_features", "mi_features", "chi_features", "pearson_features", "сlassic_GA", "optuna_features", "horizon_selected_features",]
trajectory = ["baseline", "engineered_datetime_features", "mi_features", "spearman_features", "chi_features", "pearson_features", "сlassic_GA", "optuna_features", "horizon_selected_features",]


trajectory_colors = {
    "baseline": "#4E79A7",
    "engineered_datetime_features": "#F28E2B",
    "mi_features": "#E15759",
    "spearman_features": "#76B7B2",
    "chi_features": "#59A14F",
    "pearson_features": "#EDC948",
    "сlassic_GA": "#B07AA1",
    "optuna_features": "#FF9DA7",
    "horizon_selected_features": "#9C755F",
}


results_path_init = "/Users/nikitasavvin/Downloads/export_prod_v2"
results_path = f"{results_path_init}/results"
article_materials_path = os.path.join(results_path_init, "article_materials")
winner_charts_dir = os.path.join(article_materials_path, "winner_charts")
os.makedirs(article_materials_path, exist_ok=True)
os.makedirs(winner_charts_dir, exist_ok=True)


def create_winner_rank_charts(df, winner_charts_dir):
    os.makedirs(winner_charts_dir, exist_ok=True)

    metrics_higher_better = {"r2"}

    for metric in sorted(df["metric"].unique()):

        df_metric = df[df["metric"] == metric].copy()

        ascending = False if metric in metrics_higher_better else True

        df_metric["rank"] = (
            df_metric
            .groupby(["dataset_name", "model"])["value"]
            .rank(method="dense", ascending=ascending)
            .astype(int)
        )

        rank_table = (
            df_metric
            .groupby(["rank", "trajectory_cols"])
            .size()
            .unstack(fill_value=0)
        )

        cols = trajectory[::-1]
        rank_table = rank_table.reindex(columns=cols, fill_value=0)
        rank_table = rank_table.sort_index()

        ranks = rank_table.index.to_numpy()

        group_gap = 1.2
        bar_width = 0.9

        fig = go.Figure()

        tick_positions = []
        shapes = []

        x_cursor = 0.0

        spacer_added = False

        for r_idx, r in enumerate(ranks):

            row = rank_table.loc[r]

            non_zero_cols = row[row > 0].sort_values(ascending=False).index.tolist()

            if len(non_zero_cols) == 0:
                continue

            start_x = x_cursor

            if not spacer_added:
                fig.add_trace(
                    go.Bar(
                        x=[x_cursor],
                        y=[0],
                        width=bar_width,
                        marker_color="rgba(0,0,0,0)",
                        showlegend=False,
                        hoverinfo="skip"
                    )
                )
                x_cursor += 1
                spacer_added = True

            for c in non_zero_cols:

                fig.add_trace(
                    go.Bar(
                        x=[x_cursor],
                        y=[row[c]],
                        width=bar_width,
                        name=c,
                        marker_color=trajectory_colors.get(c, "#999999"),
                        text=[row[c]],
                        textposition="outside",
                        textfont=dict(size=14),
                        showlegend=(r_idx == 0)
                    )
                )

                x_cursor += 1

            end_x = x_cursor - 1

            tick_positions.append((start_x + end_x) / 2)

            if r_idx < len(ranks) - 1:
                shapes.append(
                    dict(
                        type="line",
                        x0=x_cursor - 0.5 + group_gap / 2,
                        x1=x_cursor - 0.5 + group_gap / 2,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(color="lightgray", width=2, dash="dot")
                    )
                )

            x_cursor += group_gap

        fig.update_layout(
            title=dict(
                text=f"{metric.upper()} Rank Distribution",
                x=0.5,
                xanchor="center",
                font=dict(size=26)
            ),
            barmode="overlay",
            width=1800,
            height=900,
            template="plotly_white",
            xaxis=dict(
                title=dict(text="Rank", font=dict(size=20)),
                tickvals=tick_positions,
                ticktext=[str(r) for r in ranks[:len(tick_positions)]],
                tickfont=dict(size=20),
                automargin=True
            ),
            yaxis=dict(
                title=dict(text="Count", font=dict(size=20)),
                tickfont=dict(size=16)
            ),
            shapes=shapes,
            legend=dict(
                orientation="h",
                y=-0.25,
                x=0.5,
                xanchor="center",
                font=dict(size=14)
            ),
            margin=dict(l=140, r=50, t=120, b=140)
        )

        fig.write_html(
            os.path.join(winner_charts_dir, f"{metric}_winner.html"),
            include_plotlyjs="cdn"
        )

        rank_table.to_csv(
            os.path.join(winner_charts_dir, f"{metric}_rank_table.csv")
        )


def load_and_merge(results_path):
    base = Path(results_path)
    files = [p for p in base.rglob("line_result_table.csv") if base in p.parents]
    dfs = [pd.read_csv(f) for f in files]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

import pandas as pd


def prepare_article_table(
        df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["trajectory_cols"] = pd.Categorical(
        df["trajectory_cols"],
        categories=trajectory,
        ordered=True,
    )

    index_cols = [
        "type",
        "dataset_name",
        "trajectory_cols",
        "points_to_pred",
    ]

    result = (
        df.pivot_table(
            index=index_cols,
            columns="model",
            values=metrics_to_table,
            aggfunc="first",
        )
        .swaplevel(0, 1, axis=1)
        .sort_index(axis=1, level=0)
        .reset_index()
    )

    result = result.sort_values(
        by=[
            "type",
            "dataset_name",
            "trajectory_cols",
            "points_to_pred",
        ]
    )

    new_columns = []

    for col in result.columns:
        if isinstance(col, tuple):
            if col[0] in index_cols:
                new_columns.append((col[0], ""))
            else:
                new_columns.append(col)
        else:
            new_columns.append((col, ""))

    result.columns = pd.MultiIndex.from_tuples(new_columns)

    return result


df = load_and_merge(results_path)


metrics_col = ['r2', 'mae', 'rmse', 'mape', 'smape', 'wape', 'bias', 'medae', 'nrmse']

base_col = ["type", "dataset_name", "trajectory_cols", "points_to_pred", "model"]

cols_to_select = base_col + metrics_col

df = df[cols_to_select]

metrics_to_table = ['r2', 'mae', 'mape']


df_to_table =  prepare_article_table(df=df)

df_to_table.to_csv(f"{article_materials_path}/all_metrix_table.csv")

order_list = df["trajectory_cols"].unique().tolist()

order = [x for x in trajectory if x in order_list] + [x for x in order_list if x not in trajectory]

df["trajectory_cols"] = pd.Categorical(df["trajectory_cols"], categories=order, ordered=True)

df.to_csv(f"{article_materials_path}/all_metrix.csv")


df_long = df.melt(
    id_vars=["type", "dataset_name", "trajectory_cols", "model"],
    value_vars=["mape", "r2", "mae", "rmse"],
    var_name="metric",
    value_name="value"
)

create_winner_rank_charts(df=df_long, winner_charts_dir=winner_charts_dir)

print(df_long)
print(df_long.columns)

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
mae = out[out["metric"] == "mae"].copy()
rmse = out[out["metric"] == "rmse"].copy()

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

mae_tmp = mae.melt(
    id_vars=["type", "dataset", "trajectory"],
    value_vars=model_cols,
    var_name="model",
    value_name="mae"
)

rmse_tmp = rmse.melt(
    id_vars=["type", "dataset", "trajectory"],
    value_vars=model_cols,
    var_name="model",
    value_name="rmse"
)


best = best.merge(
    r2_tmp,
    on=["type", "dataset", "trajectory", "model"],
    how="left"
).merge(
    mae_tmp,
    on=["type", "dataset", "trajectory", "model"],
    how="left"
).merge(
    rmse_tmp,
    on=["type", "dataset", "trajectory", "model"],
    how="left"
)[["type", "dataset", "trajectory", "model", "mape", "r2", "mae", "rmse"]].reset_index(drop=True)


print(best.columns.tolist())

print(best[["mae", "rmse"]].head())

best = best[
    ["type", "dataset", "trajectory", "model", "mape", "r2", "mae", "rmse"]
].reset_index(drop=True)

# print(out)
# print("="*100)
# print(best)



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