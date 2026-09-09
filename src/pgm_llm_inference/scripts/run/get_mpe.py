"""Benchmark exact MPE against topological greedy decoding.

Run with ``python -m pgm_llm_inference.scripts.run.get_mpe``.
"""

import random

import pandas as pd

from pgm_llm_inference.evaluation.greedy_mpe import run_dataset
from pgm_llm_inference.paths import DATASETS_DIR

DATASETS = [
    "adhd_cbeb.bif",
    "covid3_cbeb.bif",
    "covid1_cbeb.bif",
    "gonorrhoeae_cbeb.bif",
    "hepar2_cbeb.bif",
    "alarm_cbeb.bif",
    "foodallergy3_cbeb.bif",
    "foodallergy1_cbeb.bif",
    "diabets_cbeb.bif",
    "child_cbeb.bif",
]
N_SAMPLES = 100
EVIDENCE_RATIO = 0.07
RANDOM_SEED: int | None = 42


def build_results_table(results: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(results)
    if table.empty:
        return table
    return table.sort_values("exact_match_mean", ascending=False).reset_index(drop=True)


def to_latex_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "% Nenhum resultado para exibir."

    rows = "\n".join(
        f"    {row.dataset:<25} & {row.n_nodes:>3} & {row.n_samples:>3} & "
        f"{row.exact_match_mean * 100:>6.2f}\\% & "
        f"{row.accuracy_mean * 100:>6.2f}\\% \\\\"
        for row in table.itertuples(index=False)
    )
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{%\n"
        "  Comparação entre MPE exato (bucket elimination max-product) e \n"
        "  decodificação gulosa em ordem topológica por dataset.\n"
        "  Exact Match: fração de amostras em que todas as variáveis \n"
        "  escondidas coincidem. Accuracy: fração de variáveis individuais\n"
        "  que coincidem (média sobre as amostras).\n"
        "}\n"
        "\\label{tab:greedy_vs_exact_mpe}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Dataset & Nós & Amostras & Exact Match & Accuracy \\\\\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    results: list[dict] = []

    print("=" * 65)
    print("  Benchmark: MPE exato vs. Decodificação Gulosa Topológica")
    print("=" * 65)
    print(f"  Datasets dir  : {DATASETS_DIR}")
    print(f"  N datasets    : {len(DATASETS)}")
    print(f"  N samples     : {N_SAMPLES}")
    print(f"  Evidence ratio: {EVIDENCE_RATIO:.0%}")
    print(f"  Random seed   : {RANDOM_SEED}")
    print("=" * 65)

    for index, dataset_name in enumerate(DATASETS, start=1):
        print(f"\n{'=' * 65}")
        print(f"  [{index}/{len(DATASETS)}] {dataset_name}")
        print("=" * 65)
        result = run_dataset(
            dataset_name=dataset_name,
            datasets_dir=DATASETS_DIR,
            sample_count=N_SAMPLES,
            evidence_ratio=EVIDENCE_RATIO,
            rng=rng,
        )
        if result is None:
            print(f"\n  ✗ {dataset_name} — ignorado.")
            continue

        print(
            f"\n  ✓ {dataset_name} | amostras OK: "
            f"{result['n_samples']}/{N_SAMPLES} | "
            f"exact_match: {result['exact_match_mean']:.4f} | "
            f"accuracy: {result['accuracy_mean']:.4f}"
        )
        results.append(result)

    table = build_results_table(results)
    print(f"\n{'=' * 65}\n  TABELA FINAL  (pandas.DataFrame)\n{'=' * 65}")
    if table.empty:
        print("  Nenhum resultado disponível.")
    else:
        print(
            table.to_string(
                index=False,
                formatters={
                    "exact_match_mean": lambda value: f"{value:.4f}",
                    "accuracy_mean": lambda value: f"{value:.4f}",
                },
            )
        )

    print(f"\n{'=' * 65}\n  TABELA FINAL  (LaTeX)\n{'=' * 65}")
    print(to_latex_table(table))


if __name__ == "__main__":
    main()
