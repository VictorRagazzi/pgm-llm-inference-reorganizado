import random
from pgm_llm_inference.experiment.sampling import (
    sample_random_evidence,
    sample_query_vars,
    sample_mpe_consistent_evidence
)

def run_batch(
    *,
    network,
    prompt_types,
    evidence_sizes,
    query_sizes,
    n_trials,
    llm_fn,
    inference_mode: str,
    base_seed=42,
    max_retries_per_trial: int = 20,
):
    max_k = max(evidence_sizes)

    for prompt_type in prompt_types:
        print(f"\n🧪 Prompt type: {prompt_type}")

        seen: set[frozenset] = set()

        for trial in range(n_trials):
            # Tenta gerar um full_evidence cujos subconjuntos fatiados
            # ainda não foram vistos para nenhum k_e
            for attempt in range(max_retries_per_trial):
                rng = random.Random(base_seed + trial * 32 + attempt * 3 + 22)

                full_evidence = sample_mpe_consistent_evidence(
                    network, k=max_k, rng=rng,
                    # não passa seen aqui — o controle é feito abaixo por subconjunto
                )

                ordered_vars = list(full_evidence.items())
                subsets = {
                    k_e: frozenset(dict(ordered_vars[:k_e]).items())
                    for k_e in sorted(evidence_sizes)
                }

                # Só aceita se TODOS os subconjuntos são novos
                if all(key not in seen for key in subsets.values() if key):
                    for key in subsets.values():
                        seen.add(key)
                    break
            else:
                print(f"  ⚠ Trial {trial}: esgotou {max_retries_per_trial} tentativas, usando último resultado")

            ordered_vars = list(full_evidence.items())

            for k_e in sorted(evidence_sizes):
                evidence = dict(ordered_vars[:k_e])
                if k_e == 0 and trial > 0:
                    continue

                if inference_mode == "map":
                    for k_q in query_sizes:
                        query_vars = sample_query_vars(
                            network,
                            k=k_q,
                            forbidden_vars=set(evidence.keys()),
                            rng=rng,
                        )
                        yield {
                            "prompt_type": prompt_type,
                            "evidence": evidence,
                            "query_vars": query_vars,
                        }

                elif inference_mode == "mpe":
                    yield {
                        "prompt_type": prompt_type,
                        "evidence": evidence,
                        "query_vars": None,
                    }

                else:
                    raise ValueError(f"Unknown inference mode {inference_mode}")

def make_evidence_sizes(limit: int, sparse_threshold: int = 10) -> list[int]:
    """
    Retorna uma lista de evidence sizes para o experimento.

    - Se limit <= sparse_threshold: range denso [1..limit].
    - Se limit >  sparse_threshold: ~8 pontos esparsos linearmente espaçados,
      sempre incluindo 1 e limit.
    """
    if limit <= sparse_threshold:
        return list(range(1, limit + 1))

    # ~8 pontos igualmente espaçados entre 1 e limit
    n_points = 8
    step = max(1, (limit - 1) // (n_points - 1))
    sizes = list(range(1, limit + 1, step))

    # Garante que limit está incluído (step pode não cair exatamente nele)
    if sizes[-1] != limit:
        sizes.append(limit)

    return sizes