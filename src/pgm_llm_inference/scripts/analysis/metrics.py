"""Generate the experiment metrics report and configured plots."""

import math

import pandas as pd

from pgm_llm_inference.analysis.logs import load_logs, translate_sampling
from pgm_llm_inference.analysis.metrics import (
    build_variable_level_table,
    compute_grouped_table,
)
from pgm_llm_inference.analysis import performance_plots
from pgm_llm_inference.analysis.structural_metrics import (
    build_structural_merge,
    build_variable_structural_merge,
    compute_parents_correlation,
    compute_variable_structural_correlations,
)
from pgm_llm_inference.analysis.structure import (
    get_structure_table,
    get_variable_structure_table,
)
from pgm_llm_inference.analysis.structure_plots import (
    plot_accuracy_by_cardinality,
    plot_accuracy_by_n_parents,
    plot_error_vs_structure,
)
from pgm_llm_inference.core.config import InferenceConfig


def print_variable_structural_correlations(
    correlations: pd.DataFrame,
    evidence_sampling: str | None = None,
) -> None:
    label = (
        f" ({translate_sampling(evidence_sampling)})"
        if evidence_sampling
        else " (todos os tipos de evidência)"
    )
    width = 60
    print("=" * width)
    print(f"{'ACURÁCIA POR VARIÁVEL × ESTRUTURA' + label:^{width}}")
    print("=" * width)
    print(f"{'Métrica estrutural':<20} {'n':>5} {'rho':>9} {'p-valor':>10}")
    print("-" * width)
    for _, row in correlations.iterrows():
        if math.isnan(row["p_value"]):
            rho_text, p_text, significance = "   N/A", "      N/A", ""
        else:
            rho_text = f"{row['rho']:>9.3f}"
            p_text = f"{row['p_value']:>10.3f}"
            significance = " *" if row["p_value"] < 0.05 else ""
        print(
            f"{row['structural_metric']:<20} {row['n']:>5} "
            f"{rho_text} {p_text}{significance}"
        )
    print("-" * width)
    print("* p < 0.05  |  n < 3 -> N/A")
    print("=" * width + "\n")


def print_parents_correlation(
    result: dict[str, float | int],
    evidence_sampling: str | None = None,
) -> None:
    label = (
        f" ({translate_sampling(evidence_sampling)})"
        if evidence_sampling
        else " (todos os tipos de evidência)"
    )
    width = 60
    print("=" * width)
    print(f"{'ACURÁCIA POR VARIÁVEL × Nº DE PAIS' + label:^{width}}")
    print("=" * width)
    if math.isnan(result["p_value"]):
        print(f"n={result['n']} — amostra insuficiente para correlação")
    else:
        significance = " *" if result["p_value"] < 0.05 else ""
        print(
            f"n={result['n']}  rho={result['rho']:.3f}  "
            f"p={result['p_value']:.3f}{significance}"
        )
    print("=" * width + "\n")


def print_results_table(
    grouped: pd.DataFrame,
    evidence_length: int | None = None,
) -> None:
    data = (
        grouped
        if evidence_length is None
        else grouped[grouped["evidence_length"] == evidence_length]
    )
    if data.empty:
        suffix = f" com evidence_length = {evidence_length}" if evidence_length else ""
        print(f"Nenhum dado encontrado{suffix}")
        return

    width = 112
    suffix = (
        f"  (evidence_length = {evidence_length})"
        if evidence_length is not None
        else "  (todas as evidence_length)"
    )
    print("=" * width)
    print(f"{'RESULTADOS' + suffix:^{width}}")
    print("=" * width)
    print(
        f"{'Dataset':<18} {'Sampling':<15} {'Layout':<12} {'Ev.Ratio':>8} "
        f"{'Acc':>7} {'F1-W':>7} {'F1-M':>7} {'Kappa':>7} "
        f"{'Consenso':>9} {'Joint':>7}"
    )
    print("-" * width)

    columns = ["dataset", "evidence_sampling", "evidence_layout", "evidence_ratio"]
    for keys, rows in data.groupby(columns):
        dataset, sampling, layout, ratio = keys
        kappa = rows["kappa"].dropna()
        kappa_text = f"{kappa.mean():>6.3f}" if len(kappa) else "   N/A"
        print(
            f"{dataset:<18} {sampling:<15} {layout:<12} {ratio:>7.0f}% "
            f"{rows['accuracy'].mean() * 100:>6.1f}% "
            f"{rows['f1_weighted'].mean():>6.3f} "
            f"{rows['f1_macro'].mean():>6.3f} {kappa_text} "
            f"{rows['consensus'].mean() * 100:>8.1f}% "
            f"{rows['joint_match'].mean() * 100:>6.1f}%"
        )

    print("-" * width)
    print(
        f"{'[MÉDIA GERAL]':<18} {'':<15} {'':<12} {'':>8} "
        f"{data['accuracy'].mean() * 100:>6.1f}% "
        f"{data['f1_weighted'].mean():>6.3f} "
        f"{data['f1_macro'].mean():>6.3f} {'':>7} "
        f"{data['consensus'].mean() * 100:>8.1f}% "
        f"{data['joint_match'].mean() * 100:>6.1f}%"
    )
    print("=" * width + "\n")


def main() -> None:
    data = load_logs(InferenceConfig().log_file_name)
    grouped = compute_grouped_table(data)

    # Switches de visualização usados durante os experimentos:
    # performance_plots.plot_accuracy_by_sampling(grouped)
    # performance_plots.plot_accuracy_vs_joint_match(grouped)
    performance_plots.plot_accuracy_by_evidence_ratio(
        grouped, evidence_sampling="mpe_consistent"
    )
    performance_plots.plot_accuracy_by_dataset(
        grouped, evidence_sampling="mpe_consistent"
    )
    performance_plots.plot_accuracy_by_dataset(
        grouped, evidence_sampling="mpe_inconsistent"
    )
    performance_plots.plot_joint_match_by_dataset(
        grouped, evidence_sampling="mpe_consistent"
    )
    performance_plots.plot_joint_match_by_dataset(
        grouped, evidence_sampling="mpe_inconsistent"
    )
    performance_plots.plot_sampling_comparison(grouped)

    print_results_table(grouped, evidence_length=1)
    print_results_table(grouped)

    datasets = data["dataset"].unique().tolist()
    merged = build_structural_merge(grouped, get_structure_table(datasets))
    plot_error_vs_structure(merged, error_metric="accuracy")
    plot_error_vs_structure(merged, error_metric="joint_match")

    variable_table = build_variable_level_table(data)
    variable_merged = build_variable_structural_merge(
        variable_table,
        get_variable_structure_table(datasets),
    )
    print_parents_correlation(compute_parents_correlation(variable_merged))
    print_variable_structural_correlations(
        compute_variable_structural_correlations(variable_merged)
    )
    plot_accuracy_by_n_parents(variable_merged)
    plot_accuracy_by_cardinality(variable_merged)


if __name__ == "__main__":
    main()
