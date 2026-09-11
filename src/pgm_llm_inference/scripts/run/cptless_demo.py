"""Demonstração prática de MPE semântico em uma rede sem CPTs.

Para trocar o domínio da aplicação, edite somente NETWORK_NAME,
VARIABLE_STATES, CHILDREN e EVIDENCE abaixo. A rede precisa ser um DAG e cada
evidência deve usar exatamente um dos estados declarados para sua variável.
"""

from pgm_llm_inference import BayesianNetwork
from pgm_llm_inference.experiment.llm_factory import build_llm_fn
from pgm_llm_inference.mpe import compile_semantic_messages, infer_from_compiled
from pgm_llm_inference.mpe.graph import topological_order


# ---------------------------------------------------------------------------
# 1. Esta é toda a definição da aplicação: nome, variáveis, arestas e evidências.
# ---------------------------------------------------------------------------

NETWORK_NAME = "Sea ice radiative forcing"

VARIABLE_STATES = {
    "Total_Cloud_Cover": ("low", "medium", "high"),
    "Total_Cloud_Water_Path": ("low", "medium", "high"),
    "Next_Longwave_flux_at_the_surface": ("low", "medium", "high"),
    "Next_Shortwave_flux_at_the_surface": ("low", "medium", "high"),
    "Sea_Ice_Concentration": ("low", "medium", "high"),
}

# Representação pai -> filhos. Nós folha podem ser omitidos.
CHILDREN = {
    "Total_Cloud_Cover": ("Next_Shortwave_flux_at_the_surface", "Next_Longwave_flux_at_the_surface"),
    "Total_Cloud_Water_Path": ("Next_Shortwave_flux_at_the_surface", "Next_Longwave_flux_at_the_surface"),
    "Next_Shortwave_flux_at_the_surface": ("Sea_Ice_Concentration",),
    "Next_Longwave_flux_at_the_surface": ("Sea_Ice_Concentration",),
}

# Variáveis observadas. Use {} para um MPE sem evidência.
EVIDENCE = {
    "Total_Cloud_Cover": "high",
}

# True usa PGM_OPENAI_* do .env; False usa PGM_LOCAL_*.
USE_REAL_LLM = False

def print_result(compiled, evidence, inferred, confidence) -> None:
    """Print the complete semantic MPE assignment as a compact table."""
    complete_assignment = {**evidence, **inferred}
    rows: list[tuple[str, str, str, str]] = []

    for variable in topological_order(compiled.bn):
        metadata = compiled.metadata.get(variable)
        display_name = (
            metadata.display_name if metadata and metadata.display_name else variable
        )
        if variable in evidence:
            origin = "evidência"
            certainty = "fixa"
        else:
            origin = "LLM-MPE"
            selected = confidence.get(variable, [None, "-"])
            certainty = str(selected[1] or "-")
        rows.append((display_name, complete_assignment[variable], origin, certainty))

    headers = ("Variável", "Valor escolhido", "Origem", "Confiança")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def separator(fill: str = "-") -> str:
        return "+" + "+".join(fill * (width + 2) for width in widths) + "+"

    def formatted_row(values: tuple[str, str, str, str]) -> str:
        cells = [f" {value:<{widths[index]}} " for index, value in enumerate(values)]
        return "|" + "|".join(cells) + "|"

    print("\n" + "=" * 72)
    print(f"MPE SEMÂNTICO (SEM CPTs) - {compiled.bn.name}")
    print("=" * 72)
    print(separator())
    print(formatted_row(headers))
    print(separator("="))
    for row in rows:
        print(formatted_row(row))
    print(separator())
    print("\nNota: este é um MPE semântico aproximado, não um MPE numérico exato.")


def main() -> None:
    network = BayesianNetwork.from_structure(
        name=NETWORK_NAME,
        variable_states=VARIABLE_STATES,
        children=CHILDREN,
    )

    llm_fn = build_llm_fn(
        use_real_llm=USE_REAL_LLM,
        use_local_llm=not USE_REAL_LLM,
    )

    # The LLM creates semantic metadata and one decision table per node.
    # No BIF file and no numerical factor/CPT are used.
    compiled = compile_semantic_messages(
        network=network,
        llm_fn=llm_fn,
        use_real_llm=USE_REAL_LLM,
        max_context_rows_per_call=32,
    )
    inferred, confidence, _ = infer_from_compiled(
        compiled=compiled,
        evidence=EVIDENCE,
    )

    print_result(compiled, EVIDENCE, inferred, confidence)


if __name__ == "__main__":
    main()
