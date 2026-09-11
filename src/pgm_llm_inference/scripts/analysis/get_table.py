"""Print structural summaries for the configured Bayesian networks.

Run with ``python -m pgm_llm_inference.scripts.analysis.get_table``.
"""

from collections.abc import Sequence
from typing import Any

from pgm_llm_inference.analysis.structure import (
    compute_parent_cardinality_correlation,
    get_structure_table,
    get_variable_structure_table,
)


def print_parent_cardinality_correlation(result: dict[str, float | int]) -> None:
    print("\n" + "=" * 65)
    print("  NÚMERO DE PAIS × CARDINALIDADE (SPEARMAN)")
    print("=" * 65)
    if result["p_value"] != result["p_value"]:
        print(
            f"  n={result['n']} | correlação indisponível "
            "(amostra/variação insuficiente)"
        )
        return

    rho = result["rho"]
    p_value = result["p_value"]
    strength = "forte" if abs(rho) >= 0.7 else "não forte"
    significance = "significativa" if p_value < 0.05 else "não significativa"
    print(f"  n={result['n']} | rho={rho:.3f} | p={p_value:.3g}")
    print(
        f"  Associação {strength} e {significance} "
        "(limiar de significância: 0,05)."
    )


def to_latex_table(results: Sequence[dict[str, Any]]) -> str:
    rows = "\n".join(
        f"{row['label']:<20} & {row['nodes']:>2} & {row['edges']:>3} & "
        f"{row['max_degree']:>1} & {row['depth']:>2} & "
        f"{row.get('mean_cardinality', float('nan')):>4.1f} & "
        f"{row.get('max_cardinality', 0):>2} \\\\"
        for row in results
    )
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Características das redes bayesianas utilizadas.}\n"
        "\\label{tab:dataset}\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\hline\n"
        "Domínio & Nós & Arestas & Grau máx. & Prof. "
        "& Card. média & Card. máx. \\\\\n"
        "\\hline\n"
        f"{rows}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def main() -> None:
    datasets = [
        "adhd_cbeb.bif",
        "alarm_cbeb.bif",
        "child_cbeb.bif",
        "diabets_cbeb.bif",
        "gonorrhoeae_cbeb.bif",
        "hepar2_cbeb.bif",
        "foodallergy1_cbeb.bif",
        "foodallergy3_cbeb.bif",
        "covid1_cbeb.bif",
        "covid3_cbeb.bif",
    ]
    table_results = get_structure_table(datasets)
    variable_results = get_variable_structure_table(datasets)
    print("\n" + to_latex_table(table_results))
    print_parent_cardinality_correlation(
        compute_parent_cardinality_correlation(variable_results)
    )


if __name__ == "__main__":
    main()
