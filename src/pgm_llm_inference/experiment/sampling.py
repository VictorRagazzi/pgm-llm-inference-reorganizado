import random
from pgm_llm_inference.models import BayesianNetwork
from pgm_llm_inference.experiment.experiment import run_max_product

def sample_random_evidence(
    network: BayesianNetwork,
    k: int,
    forbidden_vars: set[str] | None = None,
    rng: random.Random | None = None,
    min_log_prob: float = -10.0,
    max_attempts: int = 100,
) -> dict[str, str]:
    """
    Sample k random evidence variables with random values from their domains.
    Resamples until the evidence is internally consistent according to the
    network's joint probability (or until max_attempts is reached).
    """
    rng = rng or random # type: ignore
    forbidden_vars = forbidden_vars or set()

    candidates = [
        v for v in network.variables.values()
        if v.name not in forbidden_vars
    ]
    k = min(k, len(candidates))

    for attempt in range(max_attempts):
        chosen_vars = rng.sample(candidates, k)
        evidence = {
            v.name: rng.choice(v.domain)
            for v in chosen_vars
        }

        log_prob = compute_log_joint(network, evidence)

        if log_prob >= min_log_prob:
            if attempt > 0:
                print(f"⚠️  Resampled evidence {attempt+1} time(s) — log_prob={log_prob:.2f}")
            return evidence

    # fallback: retorna o último mesmo que inconsistente
    print(f"⚠️  Could not sample consistent evidence after {max_attempts} attempts. Using last sample.")
    return evidence

def sample_query_vars(
    network: BayesianNetwork,
    k: int,
    forbidden_vars: set[str],
    rng: random.Random | None = None,
) -> list[str]:
    """
    Sample k query variables not in evidence.
    """
    rng = rng or random

    candidates = [
        v.name for v in network.variables.values()
        if v.name not in forbidden_vars
    ]

    k = min(k, len(candidates))
    return rng.sample(candidates, k)


def compute_log_joint(
    network: BayesianNetwork,
    evidence: dict[str, str],
) -> float:
    """
    Computes log P(evidence) by summing log P(var | parents)
    for each observed variable whose parents are all also observed.
    Skips variables with unobserved parents (partial evaluation).
    Returns -inf if any evaluated probability is zero.
    """
    import math

    log_prob = 0.0

    for var_name, value in evidence.items():
        factor = next(
            (f for f in network.factors if f.scope[0].name == var_name),
            None,
        )
        if factor is None:
            continue

        parent_vars = factor.scope[1:]

        # Pula se algum pai não está na evidência
        if not all(p.name in evidence for p in parent_vars):
            continue

        # Índice do valor do filho
        child_var = factor.scope[0]
        try:
            child_idx = child_var.domain.index(value)
        except ValueError:
            return float("-inf")

        # Índices dos pais
        parent_indices = []
        for p in parent_vars:
            try:
                parent_indices.append(p.domain.index(evidence[p.name]))
            except ValueError:
                return float("-inf")

        # Acessa o valor na CPT: shape é (child, parent1, parent2, ...)
        idx = tuple([child_idx] + parent_indices)
        prob = float(factor.values[idx])

        if prob <= 0:
            return float("-inf")

        log_prob += math.log(prob)

    return log_prob


def sample_mpe_consistent_evidence(
    network,
    k: int,
    forbidden_vars: set[str] | None = None,
    rng: random.Random | None = None,
    bias_toward: str = "leaves",  # "roots" | "leaves" | "none"
    bias_prob: float = 1,
    max_retries: int = 10,  # <-- novo parâmetro
    seen: set[frozenset] | None = None,  # <-- conjunto compartilhado entre chamadas
) -> dict[str, str]:
    seen = seen if seen is not None else set()

    for _ in range(max_retries):
        result = _sample_once(network, k, forbidden_vars, rng, bias_toward, bias_prob)
        key = frozenset(result.items())
        if key not in seen:
            seen.add(key)
            return result

    # Se esgotar retries, retorna mesmo assim (melhor que travar)
    return result

def _sample_once(
    network,
    k: int,
    forbidden_vars: set[str] | None = None,
    rng: random.Random | None = None,
    bias_toward: str = "roots",   # padrão agora é "roots"
    bias_prob: float = 1.0,
) -> dict[str, str]:
    rng = rng or random
    forbidden_vars = forbidden_vars or set()

    candidates = [
        v for v in network.variables.values()
        if v.name not in forbidden_vars
    ]
    k = min(k, len(candidates))

    # Calcula profundidades uma vez (pode ser cacheado externamente se necessário)
    depths = compute_topological_depths(network)

    # Ordena por (profundidade, ruído) — raízes primeiro, desempate aleatório
    candidates_sorted = sorted(
        candidates,
        key=lambda v: (depths.get(v.name, 0), rng.random())
    )

    # MPE incondicional para pegar valores coerentes
    mpe_unconditional = run_max_product(
        network=network,
        query_vars=[v.name for v in candidates_sorted],
        evidence={},
    )

    selected = candidates_sorted[:k]
    return {
        var.name: mpe_unconditional["map_assignment"][var.name]
        for var in selected
    }

def compute_topological_depths(network) -> dict[str, int]:
    """Retorna a profundidade topológica de cada variável (0 = raiz sem pais)."""
    depths = {}
    
    def depth_of(var_name: str) -> int:
        if var_name in depths:
            return depths[var_name]
        var = network.variables[var_name]
        parents = list(var.parents) if hasattr(var, 'parents') else []
        if not parents:
            depths[var_name] = 0
        else:
            depths[var_name] = 1 + max(depth_of(p) for p in parents)
        return depths[var_name]
    
    for name in network.variables:
        depth_of(name)
    return depths