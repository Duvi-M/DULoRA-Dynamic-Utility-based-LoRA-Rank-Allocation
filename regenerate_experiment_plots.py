import argparse
from pathlib import Path

import pandas as pd

from src.plots import (
    plot_allocation_history_from_dataframe,
    plot_layer_scores_from_dataframe,
    plot_rank_pattern_from_dataframe,
)


def regenerate_plots(output_dir: Path):
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    layer_scores_path = output_dir / "layer_scores.csv"
    rank_pattern_path = output_dir / "rank_pattern.csv"
    allocation_history_path = output_dir / "allocation_history.csv"

    if layer_scores_path.exists():
        layer_scores_df = pd.read_csv(layer_scores_path)
        plot_layer_scores_from_dataframe(layer_scores_df, str(output_dir))
        print(f"Regenerated layer score plots for: {output_dir}")
    else:
        print(f"Warning: missing {layer_scores_path}; skipping layer score plots")

    if rank_pattern_path.exists():
        rank_pattern_df = pd.read_csv(rank_pattern_path)
        plot_rank_pattern_from_dataframe(rank_pattern_df, str(output_dir))
        print(f"Regenerated rank pattern plots for: {output_dir}")
    else:
        print(f"Warning: missing {rank_pattern_path}; skipping rank pattern plots")

    if allocation_history_path.exists():
        allocation_history_df = pd.read_csv(allocation_history_path)
        plot_allocation_history_from_dataframe(allocation_history_df, str(output_dir))
        print(f"Regenerated budget history plot for: {output_dir}")
    else:
        print(
            f"Warning: missing {allocation_history_path}; "
            "skipping budget history plot"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate experiment plots from existing CSV outputs."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Experiment output directory containing CSV files.",
    )
    args = parser.parse_args()

    regenerate_plots(Path(args.output_dir))


if __name__ == "__main__":
    main()
