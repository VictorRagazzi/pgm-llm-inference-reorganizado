"""Inspect domain and top-token uncertainty from a trusted compiled pickle."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from pgm_llm_inference.analysis.domain_entropy import compiled_row_entropy_table
from pgm_llm_inference.analysis.entropy_plots import plot_compiled_row_entropy
from pgm_llm_inference.mpe.compile import COMPILED_SCHEMA_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Pickle can execute code on load; use only artifacts generated locally.
    with args.compiled.open("rb") as file:
        compiled = pickle.load(file)
    if getattr(compiled, "schema_version", None) != COMPILED_SCHEMA_VERSION:
        raise ValueError(
            f"Compiled schema {getattr(compiled, 'schema_version', None)!r}; "
            f"expected {COMPILED_SCHEMA_VERSION}. Recompile to capture token scores."
        )

    table = compiled_row_entropy_table(compiled)
    if table.empty:
        raise ValueError("Compiled artifact has no decision rows.")
    csv_path = args.output_dir / "decision_entropy.csv"
    plot_path = args.output_dir / "decision_entropy.png"
    for path in (csv_path, plot_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    plot_compiled_row_entropy(table, plot_path)
    print(f"{len(table)} context rows; CSV: {csv_path}; plot: {plot_path}")


if __name__ == "__main__":
    main()
