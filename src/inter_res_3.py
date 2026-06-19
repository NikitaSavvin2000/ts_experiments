"""
pdm run src/inter_res_3.py
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
from plotly.subplots import make_subplots

# trajectory = ["baseline", "calendar_components",  "engineered_datetime_features", "stat_selected_features", "optuna", "сlassic_GA", "GA_horizon_selected_features"]
trajectory = ["baseline", "engineered_datetime_features", "mi_features", "pearson_features", "сlassic_GA", "optuna_features", "horizon_selected_features",]
trajectory = ["baseline", "engineered_datetime_features", "mi_features", "spearman_features", "pearson_features", "сlassic_GA", "optuna_features", "horizon_selected_features",]
trajectories = trajectory


# trajectory_colors = {
#     "baseline": "#4E79A7",
#     "engineered_datetime_features": "#F28E2B",
#     "mi_features": "#E15759",
#     "spearman_features": "#76B7B2",
#     "pearson_features": "#EDC948",
#     "сlassic_GA": "#B07AA1",
#     "optuna_features": "#FF9DA7",
#     "horizon_selected_features": "#9C755F",
# }

trajectory_colors = {
    "baseline": "#A0A0A0",
    "engineered_datetime_features": "#4E79A7",
    "mi_features": "#E15759",
    "spearman_features": "#F28E2B",
    "pearson_features": "#EDC948",
    "сlassic_GA": "#B07AA1",
    "optuna_features": "#76B7B2",
    "horizon_selected_features": "#59A14F",
}
russian_mood = True
# russian_mood = False


# new_datasets = ["morocco_zone_1", "russia_amur_region", "Istanbul_Traffic_Index", "Metro_Traffic", "Weather", "Weather_water_temp", "russia_amur_region", "Temperature_in_Celsius"]
new_datasets = ["russia_amur_region", "Istanbul_Traffic_Index", "Metro_Traffic", "Temperature_in_Celsius"]

russian_vers = [ "Istanbul_Traffic_Index", "Daily_Climate"]

bad = ["morocco_zone_1", "russia_elista", "Daily_Climate"]
bad = ["morocco_zone_1",]


russian_vers = ["morocco_zone_1", "russia_elista", "Istanbul_Traffic_Index", "Metro_Traffic", "Weather", "Temperature_in_Celsius"]
russian_vers = ["Daily_Climate", "Istanbul_Traffic_Index", "Weather"]
russian_vers = ["morocco_zone_1",]

russian_vers = ["Temperature_in_Celsius", "Daily_Climate", "Istanbul_Traffic_Index", "russia_amur_region"]
russian_vers = ["Temperature_in_Celsius","russia_amur_region"]

good = ["Temperature_in_Celsius", "Daily_Climate", "Istanbul_Traffic_Index", ]
# russian_vers = ["morocco_zone_1"]
# russian_vers = ["morocco_zone_1",]



results_path_init = "/Users/nikitasavvin/Downloads/export_prod_run"
results_path = f"{results_path_init}/results"
article_materials_path = os.path.join(results_path_init, "article_materials")
winner_charts_dir = os.path.join(article_materials_path, "winner_charts")
os.makedirs(article_materials_path, exist_ok=True)
os.makedirs(winner_charts_dir, exist_ok=True)
import os
import plotly.graph_objects as go
import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd


def create_winner_rank_charts(df, winner_charts_dir):
    import os
    import plotly.graph_objects as go

    os.makedirs(winner_charts_dir, exist_ok=True)

    metrics_higher_better = {"r2"}

    for metric in sorted(df["metric"].unique()):
        df_metric = df[df["metric"] == metric].copy()

        ascending = metric not in metrics_higher_better

        df_metric["rank"] = (
            df_metric
            .groupby(["dataset_name", "model", "points_to_pred"])["value"]
            .rank(method="dense", ascending=ascending)
            .astype(int)
        )

        rank_table = (
            df_metric
            .groupby(["rank", "trajectory_cols", "points_to_pred"])
            .size()
            .groupby(level=["rank", "trajectory_cols"])
            .sum()
            .unstack(fill_value=0)
        )

        cols = trajectory[::-1]
        rank_table = rank_table.reindex(columns=cols, fill_value=0).sort_index()

        ranks = rank_table.index.to_numpy()

        fig = go.Figure()

        tick_positions = []
        shapes = []

        x_cursor = 0.0
        spacer_added = False
        group_gap = 1.2
        bar_width = 0.9

        for r_idx, r in enumerate(ranks):
            row = rank_table.loc[r]
            non_zero_cols = row[row > 0].sort_values(ascending=False).index.tolist()

            if not non_zero_cols:
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


def create_metric_bar_chart(df, trajectories, metric, winner_charts_dir):

    os.makedirs(winner_charts_dir, exist_ok=True)

    df = df[df["trajectory_cols"].isin(trajectories)].copy()

    agg = (
        df.groupby("trajectory_cols")[metric]
        .median()
        .reindex(trajectories)
        # .dropna()
    )

    agg_df = agg.reset_index(name="median")
    agg_df = agg_df.sort_values("median", ascending=False).reset_index(drop=True)

    agg_df["main_label"] = agg_df["median"].apply(lambda x: f"{x:.3f} % {metric}")

    agg_df["color"] = agg_df["trajectory_cols"].map(trajectory_colors)

    for i in range(1, len(agg_df)):


        row_past = agg_df.iloc[i-1]
        row_act = agg_df.iloc[i]

        past_median = row_past["median"]
        actual_median = row_act["median"]

        add_color = row_past["color"]

        diff = past_median - actual_median
        pct = round((past_median - actual_median) / past_median * 100, 1)

        agg_df.loc[i, f"add_value_{i}"] = diff
        agg_df.loc[i, f"add_label_{i}"] = f"←  - {pct}%"
        agg_df.loc[i, f"add_color_{i}"] = add_color

        print(f"past_median = {past_median}")
        print(f"actual_median = {actual_median}")

    agg_df = agg_df.replace(["NaN", "nan", "NAN", None, ""], np.nan)
    agg_df = agg_df.apply(pd.to_numeric, errors="ignore")

    agg_df = agg_df.fillna(method="ffill")

    add_ids = sorted(
        {
            int(c.split("_")[-1])
            for c in agg_df.columns
            if c.startswith("add_value_")
        },
        reverse=True
    )

    fig = go.Figure()
    fig.update_layout(
        height=1200
    )

    x = agg_df["trajectory_cols"].tolist()

    fig.add_trace(
        go.Bar(
            x=x,
            y=agg_df["median"].fillna(0).values,
            marker_color=agg_df["color"].values,
            text=agg_df["main_label"].values,
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(
                size=18,
                color="black",
                family="Arial"
            ),
            name=""
        )
    )

    for i in range(len(x) + 1):
        fig.add_shape(
            type="line",
            x0=i - 0.5,
            x1=i - 0.5,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(
                color="rgba(0,0,0,0.2)",
                width=1
            )
        )

    for k in add_ids:
        vcol = f"add_value_{k}"
        lcol = f"add_label_{k}"
        ccol = f"add_color_{k}"

        label = agg_df[lcol].fillna("").values

        if vcol not in agg_df.columns:
            continue

        fig.add_trace(
            go.Bar(
                x=x,
                y=agg_df[vcol].fillna(0).values,
                marker=dict(
                    color=agg_df[ccol].fillna("#000000").values,
                    pattern=dict(
                        shape="/"
                    )
                ),
                text=label,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(
                    size=18,
                    color="black",
                    family="Arial"
                ),
                name=f"add_{k}",
                showlegend=False
            )
        )

    fig.update_layout(
        xaxis=dict(
            title="Траектории",
            showgrid=True
        ),
        yaxis=dict(
            title=f"Среднее начение метрики ({metric}, %)",
            showgrid=True
        ),
        barmode="stack",
        template="none",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    path = os.path.join(winner_charts_dir, f"{metric}.html")
    fig.write_html(path)

    return agg_df


def load_and_merge(results_path):
    from pathlib import Path
    import pandas as pd

    base = Path(results_path)
    files = [p for p in base.rglob("line_result_table.csv") if base in p.parents]

    dfs = []
    broken_files = []

    for f in files:
        if any(b in str(f) for b in bad):
            continue
        try:
            df = pd.read_csv(f)
            dfs.append(df)

        except Exception:
            broken_files.append(str(f))

            try:
                bad_df = pd.read_csv(f, engine="python")
                print("\n❌ BROKEN FILE:")
                print(f)
                print("\n📌 columns:")
                print(list(bad_df.columns))
                print("\n📌 shape:")
                print(bad_df.shape)

            except Exception:
                print("\n❌ BROKEN FILE (unreadable):")
                print(f)

    if broken_files:
        print("\n====================")
        print("❌ TOTAL BROKEN FILES:")
        for f in broken_files:
            print(f)
        print("====================\n")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


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

# df = df[df["dataset_name"].isin(new_datasets)]

if russian_mood:
    df = df[df["dataset_name"].isin(russian_vers)]


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
    id_vars=["type", "dataset_name", "trajectory_cols", "points_to_pred", "model"],
    value_vars=["mape", "r2", "mae", "rmse"],
    var_name="metric",
    value_name="value"
)

create_winner_rank_charts(df=df_long, winner_charts_dir=winner_charts_dir)


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


# trajectories = ["baseline", "engineered_datetime_features", "horizon_selected_features",]
metric = "mape"
# metric = "mae"


create_metric_bar_chart(df=df, trajectories=trajectories, metric=metric, winner_charts_dir=winner_charts_dir)



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

list_metrics = ["mae", "mape"]

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