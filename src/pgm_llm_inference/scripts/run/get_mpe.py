"""
scripts/run/get_mpe.py
======================
Benchmark: MPE exato (bucket elimination max-product, via infraestrutura
interna do projeto) vs. decodificação gulosa em ordem topológica.

Fluxo por dataset (.bif):
  1. Carrega a rede com `load_network` (retorna BayesianNetwork interna).
  2. Gera N_SAMPLES configurações de evidência aleatória
     (proporção EVIDENCE_RATIO das variáveis, sorteadas uniformemente).
  3. Para cada amostra:
       a. MPE exato via `run_max_product` (MaxProductStrategy + VE numérico).
       b. Decodificação gulosa em ordem topológica:
            para cada variável não-observada xi, escolhe
               argmax_{xi} P(xi | pa(Xi))
            usando apenas os valores já fixados dos pais (evidência ou
            decisão gulosa anterior) — sem olhar para filhos/descendentes.
       c. Compara:
            • exact_match = 1 se TODAS as variáveis coincidem, 0 c.c.
            • accuracy    = fração de variáveis individuais que coincidem.
  4. Agrega exact_match e accuracy médios por dataset.

Saída final:
  • pandas.DataFrame ordenado por exact_match decrescente.
  • Tabela LaTeX (estilo booktabs).

Uso:
    # A partir da raiz do projeto:
    python scripts/run/get_mpe.py

    # Ou via módulo:
    python -m pgm_llm_inference.scripts.run.get_mpe
"""
from __future__ import annotations

import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Garante que a raiz do projeto está no sys.path para execução standalone
# (python scripts/run/get_mpe.py) sem necessidade de `pip install -e .`.
#
# A heurística sobe a árvore de diretórios a partir do próprio arquivo até
# encontrar a raiz do repositório (identificada pela presença do pacote
# `pgm_llm_inference`). Funciona independentemente de onde o script está
# instalado ou de como é invocado.
# ---------------------------------------------------------------------------
def _find_project_root(start: Path) -> Path:
    """
    Sobe a árvore de diretórios a partir de `start` até encontrar um
    diretório que contenha o pacote `pgm_llm_inference` (ou o próprio
    arquivo `pgm_llm_inference/__init__.py`). Se não encontrar, retorna
    o diretório pai do script como fallback.
    """
    candidate = start.resolve()
    for _ in range(10):  # no máximo 10 níveis acima
        if (candidate / "pgm_llm_inference").is_dir():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    # Fallback: assume que o script está em scripts/run/ e sobe 3 níveis
    try:
        return start.resolve().parents[3]
    except IndexError:
        return start.resolve().parent


PROJECT_ROOT = _find_project_root(Path(__file__))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports da infraestrutura interna do projeto
# ---------------------------------------------------------------------------
from pgm_llm_inference.io.loaders import load_network          # carrega .bif → BayesianNetwork interna
from pgm_llm_inference.models import BayesianNetwork           # tipo retornado por load_network
from pgm_llm_inference.experiment.experiment import (
    run_max_product,   # MPE exato: MaxProductStrategy + VE numérico
    get_hidden_vars,   # retorna variáveis não-observadas
)
from pgm_llm_inference.mpe.graph import topological_order      # ordem topológica sobre BayesianNetwork interna


# ===========================================================================
# PARÂMETROS CONFIGURÁVEIS
# ===========================================================================

# Diretório onde estão os arquivos .bif
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"

# Lista de datasets (.bif) a avaliar.
# Datasets marcados com "# ERRO" falham no carregamento — o script os ignora
# e continua automaticamente.
DATASETS: List[str] = [
    "adhd_cbeb.bif",
    "covid3_cbeb.bif",        # ERRO esperado (manter para testar robustez)
    "covid1_cbeb.bif",        # ERRO esperado (manter para testar robustez)
    "gonorrhoeae_cbeb.bif",
    "hepar2_cbeb.bif",
    "alarm_cbeb.bif",
    "foodallergy3_cbeb.bif",
    "foodallergy1_cbeb.bif",
    "diabets_cbeb.bif",
    "child_cbeb.bif",
]

# Número de amostras de evidência por dataset
N_SAMPLES: int = 100

# Proporção de variáveis observadas (evidência) em cada amostra, em [0, 1).
#
# Nota: com EVIDENCE_RATIO baixo, o número de variáveis escondidas é alto e
# o MPE exato pode ficar lento ou consumir muita memória em redes densas
# (ex.: hepar2, alarm). Aumente EVIDENCE_RATIO (p. ex., 0.30–0.50) se
# necessário para manter a inferência tratável.
EVIDENCE_RATIO: float = 0.07

# Semente aleatória para reprodutibilidade (None = não fixa)
RANDOM_SEED: Optional[int] = 42


# ===========================================================================
# CARREGAMENTO DE REDE
# ===========================================================================

def load_bif_network(path: Path) -> BayesianNetwork:
    """
    Carrega uma rede Bayesiana de um arquivo .bif usando o loader interno do
    projeto, que converte o modelo pgmpy em BayesianNetwork interna
    (com Variables e Factors numéricos, pronto para inferência via VE).

    Retorna a BayesianNetwork interna (pgm_llm_inference.models.BayesianNetwork).
    """
    network, _ctx = load_network(str(path))
    return network


# ===========================================================================
# GERAÇÃO DE EVIDÊNCIA ALEATÓRIA
# ===========================================================================

def sample_random_evidence(
    network: BayesianNetwork,
    ratio: float,
    rng: random.Random,
) -> Dict[str, str]:
    """
    Sorteia um subconjunto de variáveis (~ratio * n_vars, com pelo menos 0
    e no máximo n_vars - 1 variáveis, garantindo ao menos uma variável
    escondida) e atribui a cada uma um estado sorteado uniformemente.

    Parâmetros
    ----------
    network : BayesianNetwork interna (retornada por load_bif_network).
    ratio   : proporção de variáveis observadas, em [0, 1).
    rng     : gerador aleatório (para reprodutibilidade).

    Retorna
    -------
    dict {var_name: state_str}
    """
    variables = list(network.variables.keys())
    n_vars = len(variables)

    n_evidence = int(round(ratio * n_vars))
    n_evidence = max(0, min(n_evidence, n_vars - 1))  # ao menos 1 variável escondida

    evidence_vars = rng.sample(variables, n_evidence)

    evidence: Dict[str, str] = {}
    for var_name in evidence_vars:
        # Variable.states é tuple[str, ...]; .domain retorna list[str]
        states = network.variables[var_name].states
        evidence[var_name] = rng.choice(states)

    return evidence


# ===========================================================================
# MPE EXATO
# ===========================================================================

def exact_mpe(
    network: BayesianNetwork,
    hidden_vars: List[str],
    evidence: Dict[str, str],
) -> Dict[str, str]:
    """
    MPE exato via bucket elimination max-product (MaxProductStrategy).

    Usa `run_max_product` do módulo experiment.experiment, que internamente
    instancia InferenceEngine + MaxProductStrategy e roda variable elimination
    com rastreamento de argmax e backtracking.

    Retorna
    -------
    dict {var_name: state_str} para as variáveis em hidden_vars.
    """
    result = run_max_product(
        network=network,
        query_vars=hidden_vars,
        evidence=evidence,
    )
    # map_assignment contém o assignment conjunto ótimo (MAP/MPE)
    full_assignment: Dict[str, str] = result["map_assignment"]
    # Filtra apenas as variáveis escondidas (evidência não deve aparecer)
    return {var: full_assignment[var] for var in hidden_vars if var in full_assignment}


# ===========================================================================
# DECODIFICAÇÃO GULOSA
# ===========================================================================

def greedy_mpe(
    network: BayesianNetwork,
    hidden_vars: List[str],
    evidence: Dict[str, str],
) -> Dict[str, str]:
    """
    Decodificação gulosa em ordem topológica.

    Para cada variável não-observada xi (na ordem topológica do DAG), escolhe:

        argmax_{xi} P(xi | pa(Xi))

    usando os valores já fixados dos pais (evidência ou decisão gulosa
    anterior). Como a ordem é topológica, todos os pais de xi foram
    processados antes de xi: ou são evidência, ou já foram decididos
    greedily — garantindo que assignment[pa] esteja sempre disponível.

    Implementação
    -------------
    Cada fator CPD (Factor) na rede interna tem:
        scope[0]  → variável filho xi
        scope[1:] → variáveis pai em ordem
        values    → np.ndarray com shape (card_xi, card_pa1, card_pa2, ...)

    Para fixar os pais, iteramos os eixos de trás para frente
    (do mais interno para o mais externo) usando np.take com o índice do
    estado do pai no domínio, o que preserva os índices corretos a cada
    passo (o slice reduz a dimensão sem deslocar os eixos anteriores).

    Parâmetros
    ----------
    network     : BayesianNetwork interna.
    hidden_vars : variáveis não-observadas (não precisam estar em ordem).
    evidence    : dict {var_name: state_str} de variáveis observadas.

    Retorna
    -------
    dict {var_name: state_str} para cada variável em hidden_vars.
    """
    assignment: Dict[str, str] = dict(evidence)

    # Ordem topológica da BayesianNetwork interna (pais antes de filhos)
    topo_order: List[str] = topological_order(network)

    # Mapa rápido: var_name → Factor (CPD) onde scope[0].name == var_name
    # convert_pgmpy_model garante que cada variável tem exatamente um fator CPD
    cpd_map = {f.scope[0].name: f for f in network.factors}

    for var_name in topo_order:
        if var_name in assignment:
            continue  # variável de evidência ou já decidida

        cpd = cpd_map[var_name]
        # values.shape = (card_xi, card_pa1, card_pa2, ...)
        values = cpd.values

        # --- Reduz o tensor fixando cada eixo parental (da direita p/ esquerda) ---
        # Processar de trás para frente evita o deslocamento de índices de eixo
        # que ocorreria se removêssemos os eixos na ordem direta.
        for axis in range(len(cpd.scope) - 1, 0, -1):
            parent_var = cpd.scope[axis]
            parent_val = assignment[parent_var.name]
            # Índice do estado fixado no domínio do pai
            val_idx = parent_var.states.index(parent_val)
            # np.take(arr, idx, axis) → array com 1 dimensão a menos
            values = np.take(values, val_idx, axis=axis)

        # Após fixar todos os pais, values é 1D: [P(xi=s0|pa), P(xi=s1|pa), ...]
        best_idx = int(np.argmax(values))
        assignment[var_name] = cpd.scope[0].states[best_idx]

    return {var: assignment[var] for var in hidden_vars}


# ===========================================================================
# COMPARAÇÃO DE ASSIGNMENTS
# ===========================================================================

def compare_assignments(
    exact: Dict[str, str],
    greedy: Dict[str, str],
    hidden_vars: List[str],
) -> Tuple[int, float]:
    """
    Compara o assignment MPE exato com o assignment guloso.

    Retorna
    -------
    (exact_match, accuracy)
      exact_match : 1 se TODAS as variáveis escondidas coincidem, 0 c.c.
      accuracy    : fração de variáveis individuais que coincidem.
    """
    if not hidden_vars:
        return 1, 1.0

    hits = sum(
        1 for v in hidden_vars
        if exact.get(v) == greedy.get(v)
    )
    accuracy = hits / len(hidden_vars)
    exact_match = int(hits == len(hidden_vars))
    return exact_match, accuracy


# ===========================================================================
# LOOP PRINCIPAL POR DATASET
# ===========================================================================

def run_dataset(
    dataset_name: str,
    datasets_dir: Path,
    n_samples: int,
    evidence_ratio: float,
    rng: random.Random,
) -> Optional[dict]:
    """
    Executa n_samples comparações (MPE exato vs. guloso) para um dataset.

    Retorna um dicionário com as métricas agregadas, ou None se o
    carregamento falhar ou nenhuma amostra for bem-sucedida.

    Erros por amostra (ex.: memória insuficiente para VE exato em redes
    muito densas com poucos observados) são logados individualmente sem
    interromper o loop.
    """
    dataset_path = datasets_dir / dataset_name

    # --- Carregamento ---
    try:
        network = load_bif_network(dataset_path)
    except Exception as exc:
        print(f"[{dataset_name}] ERRO ao carregar: {exc}")
        traceback.print_exc()
        return None

    n_nodes = len(network.variables)
    print(f"  Rede carregada: {n_nodes} nós, {len(network.factors)} fatores")

    exact_matches: List[int] = []
    accuracies: List[float] = []
    n_ok = 0

    for sample_idx in range(n_samples):
        evidence: Dict[str, str] = {}
        try:
            evidence = sample_random_evidence(network, evidence_ratio, rng)
            hidden_vars = get_hidden_vars(network, evidence)

            if not hidden_vars:
                # Sem variáveis escondidas — pula a amostra (não deveria ocorrer
                # com n_evidence < n_vars, mas por segurança)
                continue

            # MPE exato (bucket elimination max-product)
            exact_assignment = exact_mpe(network, hidden_vars, evidence)

            # Decodificação gulosa em ordem topológica
            greedy_assignment = greedy_mpe(network, hidden_vars, evidence)

            exact_match, accuracy = compare_assignments(
                exact_assignment, greedy_assignment, hidden_vars
            )

            exact_matches.append(exact_match)
            accuracies.append(accuracy)
            n_ok += 1

            if (sample_idx + 1) % 10 == 0 or sample_idx == n_samples - 1:
                running_em = sum(exact_matches) / n_ok
                running_acc = sum(accuracies) / n_ok
                print(
                    f"  [{sample_idx + 1:3d}/{n_samples}] "
                    f"ok={n_ok} | "
                    f"exact_match={running_em:.3f} | "
                    f"accuracy={running_acc:.3f} | "
                    f"ev={len(evidence)} | hidden={len(hidden_vars)}"
                )

        except Exception as exc:
            ev_repr = evidence if evidence else "?"
            print(
                f"  [{dataset_name}] ERRO na amostra {sample_idx} "
                f"(evidência={ev_repr}): {exc}"
            )
            traceback.print_exc()
            continue

    if n_ok == 0:
        print(f"[{dataset_name}] Nenhuma amostra bem-sucedida — dataset ignorado.")
        return None

    return {
        "dataset": dataset_name,
        "n_nodes": n_nodes,
        "n_samples": n_ok,
        "exact_match_mean": sum(exact_matches) / n_ok,
        "accuracy_mean": sum(accuracies) / n_ok,
    }


# ===========================================================================
# TABELA DE RESULTADOS
# ===========================================================================

def build_results_table(results: List[dict]) -> pd.DataFrame:
    """
    Constrói um DataFrame com os resultados de todos os datasets, ordenado
    por exact_match_mean decrescente.
    """
    df = pd.DataFrame(results)
    if df.empty:
        return df
    df = df.sort_values(by="exact_match_mean", ascending=False).reset_index(drop=True)
    return df


def to_latex_table(df: pd.DataFrame) -> str:
    """
    Gera uma tabela LaTeX em estilo booktabs a partir do DataFrame de resultados.

    Colunas: Dataset | Nós | Amostras | Exact Match (%) | Accuracy (%)
    """
    if df.empty:
        return "% Nenhum resultado para exibir."

    rows = "\n".join(
        (
            f"    {row.dataset:<25} & {row.n_nodes:>3} & {row.n_samples:>3} & "
            f"{row.exact_match_mean * 100:>6.2f}\\% & {row.accuracy_mean * 100:>6.2f}\\% \\\\"
        )
        for row in df.itertuples(index=False)
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


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    rng = random.Random(RANDOM_SEED)
    results: List[dict] = []

    print("=" * 65)
    print("  Benchmark: MPE exato vs. Decodificação Gulosa Topológica")
    print("=" * 65)
    print(f"  Datasets dir  : {DATASETS_DIR}")
    print(f"  N datasets    : {len(DATASETS)}")
    print(f"  N samples     : {N_SAMPLES}")
    print(f"  Evidence ratio: {EVIDENCE_RATIO:.0%}")
    print(f"  Random seed   : {RANDOM_SEED}")
    print("=" * 65)

    for idx, dataset_name in enumerate(DATASETS, start=1):
        print(f"\n{'=' * 65}")
        print(f"  [{idx}/{len(DATASETS)}] {dataset_name}")
        print(f"{'=' * 65}")

        res = run_dataset(
            dataset_name=dataset_name,
            datasets_dir=DATASETS_DIR,
            n_samples=N_SAMPLES,
            evidence_ratio=EVIDENCE_RATIO,
            rng=rng,
        )

        if res is not None:
            print(
                f"\n  ✓ {dataset_name}  |  "
                f"amostras OK: {res['n_samples']}/{N_SAMPLES}  |  "
                f"exact_match: {res['exact_match_mean']:.4f}  |  "
                f"accuracy: {res['accuracy_mean']:.4f}"
            )
            results.append(res)
        else:
            print(f"\n  ✗ {dataset_name} — ignorado.")

    # --- Tabela final ---
    df = build_results_table(results)

    print(f"\n{'=' * 65}")
    print("  TABELA FINAL  (pandas.DataFrame)")
    print(f"{'=' * 65}")
    if df.empty:
        print("  Nenhum resultado disponível.")
    else:
        print(
            df.to_string(
                index=False,
                formatters={
                    "exact_match_mean": lambda x: f"{x:.4f}",
                    "accuracy_mean": lambda x: f"{x:.4f}",
                },
            )
        )

    print(f"\n{'=' * 65}")
    print("  TABELA FINAL  (LaTeX)")
    print(f"{'=' * 65}")
    print(to_latex_table(df))


if __name__ == "__main__":
    main()