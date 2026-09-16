"""Inspect domain and top-token uncertainty from a trusted compiled pickle."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from pgm_llm_inference.analysis.domain_entropy import compiled_row_entropy_table
from pgm_llm_inference.analysis.entropy_plots import (
    plot_compiled_row_entropy,
    plot_token_vector,
    token_vector_filename,
)
from pgm_llm_inference.analysis.token_vectors import compiled_token_vector_table
from pgm_llm_inference.mpe.compile import COMPILED_SCHEMA_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--token-vectors-only",
        action="store_true",
        help="Generate only the raw token-vector CSV and one SVG per variable.",
    )
    args = parser.parse_args()

    # Pickle can execute code on load; use only artifacts generated locally.
    with args.compiled.open("rb") as file:
        compiled = pickle.load(file)
    if getattr(compiled, "schema_version", None) != COMPILED_SCHEMA_VERSION:
        raise ValueError(
            f"Compiled schema {getattr(compiled, 'schema_version', None)!r}; "
            f"expected {COMPILED_SCHEMA_VERSION}. Recompile to capture token scores."
        )

    vectors = compiled_token_vector_table(compiled)
    if args.token_vectors_only and vectors.empty:
        raise ValueError(
            "This pickle has no token_scores in its context rows; "
            "raw token vectors cannot be plotted."
        )

    output_paths = []
    csv_path = args.output_dir / "decision_entropy.csv"
    plot_path = args.output_dir / "decision_entropy.png"
    if not args.token_vectors_only:
        table = compiled_row_entropy_table(compiled)
        if table.empty:
            raise ValueError("Compiled artifact has no decision rows.")
        output_paths.extend((csv_path, plot_path))

    variable_plots = (
        {
            variable: args.output_dir / "token_vectors" / token_vector_filename(variable)
            for variable in vectors["variable"].unique()
        }
        if not vectors.empty
        else {}
    )
    vector_csv = args.output_dir / "token_vectors" / "token_vectors.csv"
    if variable_plots:
        output_paths.extend((vector_csv, *variable_plots.values()))

    for path in output_paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.token_vectors_only:
        table.to_csv(csv_path, index=False)
        plot_compiled_row_entropy(table, plot_path)
        print(f"{len(table)} context rows; CSV: {csv_path}; plot: {plot_path}")
    if variable_plots:
        vector_csv.parent.mkdir(parents=True, exist_ok=True)
        vectors.to_csv(vector_csv, index=False)
        for variable, path in variable_plots.items():
            plot_token_vector(vectors, variable, path)
        print(f"{len(variable_plots)} token-vector plots; CSV: {vector_csv}")
    elif not args.token_vectors_only:
        print("No token_scores found; no token-vector plots were generated.")


if __name__ == "__main__":
    main()
