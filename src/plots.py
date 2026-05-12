import os
import re

import matplotlib.pyplot as plt
import pandas as pd


ATTENTION_COLUMNS = ["query", "value"]
MODULE_SHORT_NAMES = {
    "query": "Q",
    "value": "V",
}


def parse_layer_module_name(layer_name: str):
    match = re.search(r"layer\.(\d+).*\.([^.]+)$", layer_name)
    if not match:
        return None, None

    layer_idx = int(match.group(1))
    module_name = match.group(2)

    if module_name not in ATTENTION_COLUMNS:
        return None, None

    return layer_idx, module_name


def shorten_layer_name(layer_name: str) -> str:
    layer_idx, module_name = parse_layer_module_name(layer_name)
    if layer_idx is not None:
        return f"L{layer_idx}-{MODULE_SHORT_NAMES[module_name]}"

    parts = layer_name.split(".")
    fallback = ".".join(parts[-3:]) if len(parts) >= 3 else layer_name
    return fallback[-28:]


def _plot_bar(df, value_column: str, output_dir: str, filename: str, title: str,
              x_label: str, y_label: str):
    labels = [shorten_layer_name(layer) for layer in df["layer"]]
    x_positions = range(len(labels))
    width = max(12, len(labels) * 0.45)

    plt.figure(figsize=(width, 5))
    plt.bar(x_positions, df[value_column])
    plt.xticks(x_positions, labels, rotation=60, ha="right")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def _build_heatmap_matrix(df, value_column: str):
    parsed_rows = []
    for _, row in df.iterrows():
        layer_idx, module_name = parse_layer_module_name(row["layer"])
        if layer_idx is None:
            continue
        parsed_rows.append((layer_idx, module_name, row[value_column]))

    if not parsed_rows:
        return None

    max_layer_idx = max(layer_idx for layer_idx, _, _ in parsed_rows)
    matrix = [
        [0 for _ in ATTENTION_COLUMNS]
        for _ in range(max_layer_idx + 1)
    ]

    for layer_idx, module_name, value in parsed_rows:
        column_idx = ATTENTION_COLUMNS.index(module_name)
        matrix[layer_idx][column_idx] = value

    return matrix


def _save_heatmap(df, value_column: str, output_dir: str, filename: str,
                  title: str, colorbar_label: str, value_format: str):
    matrix = _build_heatmap_matrix(df, value_column)
    if matrix is None:
        print(f"Warning: could not parse layer names; skipping {filename}")
        return

    plt.figure(figsize=(5, max(5, len(matrix) * 0.45)))
    image = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(image, label=colorbar_label)
    plt.xticks(range(len(ATTENTION_COLUMNS)), ["Query", "Value"])
    plt.yticks(range(len(matrix)), [f"Layer {idx}" for idx in range(len(matrix))])
    plt.xlabel("Attention module")
    plt.ylabel("Transformer layer")
    plt.title(title)

    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            plt.text(
                col_idx,
                row_idx,
                value_format.format(value),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def plot_layer_scores_from_dataframe(df, output_dir: str):
    _plot_bar(
        df=df,
        value_column="score",
        output_dir=output_dir,
        filename="layer_scores.png",
        title="Layer importance scores",
        x_label="LoRA module",
        y_label="Gradient importance score",
    )
    _save_heatmap(
        df=df,
        value_column="score",
        output_dir=output_dir,
        filename="layer_scores_heatmap.png",
        title="Layer importance scores",
        colorbar_label="Gradient importance score",
        value_format="{:.2f}",
    )


def plot_rank_pattern_from_dataframe(df, output_dir: str):
    _plot_bar(
        df=df,
        value_column="rank",
        output_dir=output_dir,
        filename="rank_pattern.png",
        title="Adaptive rank distribution",
        x_label="LoRA module",
        y_label="Assigned rank",
    )
    _save_heatmap(
        df=df,
        value_column="rank",
        output_dir=output_dir,
        filename="rank_heatmap.png",
        title="Adaptive rank distribution",
        colorbar_label="Assigned rank",
        value_format="{:.0f}",
    )


def plot_allocation_history_from_dataframe(df, output_dir: str):
    if df.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df["iteration"], df["used_budget"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Used rank budget")
    plt.title("Budget allocation over iterations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "budget_history.png"), dpi=300)
    plt.close()


def save_layer_scores_plot(layer_scores: dict, output_dir: str):
    df = pd.DataFrame(
        list(layer_scores.items()),
        columns=["layer", "score"],
    )

    csv_path = os.path.join(output_dir, "layer_scores.csv")
    df.to_csv(csv_path, index=False)
    plot_layer_scores_from_dataframe(df, output_dir)


def save_rank_pattern_plot(rank_pattern: dict, output_dir: str):
    df = pd.DataFrame(
        list(rank_pattern.items()),
        columns=["layer", "rank"],
    )

    csv_path = os.path.join(output_dir, "rank_pattern.csv")
    df.to_csv(csv_path, index=False)
    plot_rank_pattern_from_dataframe(df, output_dir)


def save_allocation_history_plot(history: list, output_dir: str):
    df = pd.DataFrame(history)

    csv_path = os.path.join(output_dir, "allocation_history.csv")
    df.to_csv(csv_path, index=False)
    plot_allocation_history_from_dataframe(df, output_dir)


def save_metrics(metrics: dict, output_dir: str):
    df = pd.DataFrame([metrics])
    df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
