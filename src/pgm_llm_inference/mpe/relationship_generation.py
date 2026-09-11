"""Generate and load qualitative relationship notes."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from ..models import BayesianNetwork

# ---- Modelos ----

class VariableRelationshipNotes(BaseModel):
    notes: list[str]

class NetworkRelationshipResponse(BaseModel):
    relationships: dict[str, VariableRelationshipNotes]


# ---- Loader ----

def load_relationship_notes(
    path: Path | None,
    bn: BayesianNetwork,
) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
 
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Relationship notes JSON must be an object keyed by variable name.")
 
    result: dict[str, tuple[str, ...]] = {}
    for variable, raw in payload.items():
        if variable not in bn.variables:
            raise ValueError(f"Relationship notes references unknown variable '{variable}'.")
        notes = raw if isinstance(raw, list) else raw.get("notes", [])
        result[variable] = tuple(notes)
 
    return result
 
 
# ---- Prompt builder ----
 
def build_relationship_notes_prompt(bn: BayesianNetwork) -> str:
    # ── Preamble ──────────────────────────────────────────────────────────────
    header = (
        "You are an expert in probabilistic graphical models and Bayesian networks.\n"
        "Your task is to write *relationship notes* for each non-root variable in the "
        "network described below. These notes will later be used as reasoning guidance "
        "for an LLM that must perform belief propagation WITHOUT access to any CPTs or "
        "numerical probabilities — qualitative reasoning only.\n\n"

        "═══════════════════════════════════════════════════════\n"
        "WHAT THE NOTES MUST CAPTURE\n"
        "═══════════════════════════════════════════════════════\n"
        "For every non-root variable write ONE cohesive paragraph that covers ALL of "
        "the following, woven together (do NOT use sub-headings or bullet points):\n\n"

        "  1. CANONICAL DRIVER  — which parent is the primary activator/suppressor "
        "and in what direction.\n"
        "  2. MODULATING PARENTS — how secondary parents shift, gate, or attenuate "
        "the canonical effect; include non-obvious cross-talk (e.g. one parent "
        "suppressing another's effect rather than acting independently).\n"
        "  3. STATE RANKING RULES — explicit guidance on when each domain value "
        "(HIGH / AVG / LOW or domain-specific equivalents) should be preferred over "
        "the others, given specific parent combinations. Phrase these as preference "
        "rules, not mere possibilities (e.g. 'X=HIGH should *outrank* X=AVG when …').\n"
        "  4. DOWNSTREAM CONSISTENCY — if this variable's children exert back-pressure "
        "(i.e. the downstream context favors a particular state), explain which "
        "child signals should reinforce or override local parent pressure.\n"
        "  5. COMMON REASONING TRAPS — at least one concrete mistake an LLM is "
        "likely to make about this variable (e.g. always picking the middle state, "
        "ignoring a gating parent, conflating correlation with activation).\n\n"

        "STYLE RULES\n"
        "───────────\n"
        "• Start each note with the header:  '<Var> <- <Parent1>, <Parent2>, …:'\n"
        "• Write in precise, mechanistic language — avoid vague hedges like 'may' or "
        "'could' unless the ambiguity is itself the point.\n"
        "• Prefer active constructions: 'LOW PKA releases RAF buffering' rather than "
        "'RAF may be less buffered when PKA is low'.\n"
        "• Keep each note to 4–8 sentences. Density over length.\n"
        "• Do NOT reproduce CPT numbers or invent probabilities.\n\n"
    )

    # ── Network structure ─────────────────────────────────────────────────────
    structure_lines = [
        "═══════════════════════════════════════════════════════\n"
        "NETWORK STRUCTURE\n"
        "═══════════════════════════════════════════════════════\n"
    ]

    root_vars = []
    non_root_vars = []

    for var_name, variable in bn.variables.items():
        parents = list(bn.parents.get(var_name, []))
        if not parents:
            root_vars.append(var_name)
        else:
            non_root_vars.append(var_name)

    # Roots — listed briefly so the LLM understands the full graph
    structure_lines.append(f"Root variables (no parents): {', '.join(root_vars)}\n")

    structure_lines.append("Non-root variables (write notes for these):\n")
    for var_name in non_root_vars:
        variable   = bn.variables[var_name]
        parents    = list(bn.parents.get(var_name, []))
        children   = [v for v in bn.variables if var_name in list(bn.parents.get(v, []))]
        siblings   = _collect_siblings(bn, var_name, parents)   # shared-parent context

        structure_lines.append(f"  [{var_name}]")
        structure_lines.append(f"    Domain  : {list(variable.states)}")
        structure_lines.append(f"    Parents : {parents}")
        structure_lines.append(f"    Children: {children if children else 'none'}")
        if siblings:
            structure_lines.append(
                f"    Siblings (share ≥1 parent): {siblings}"
                " — note any competitive or complementary dynamics"
            )
        structure_lines.append("")

    SYNTHETIC_EXAMPLE = """\
        Example of the expected output style (fictitious network, do not reuse these variables):

        {
        "relationships": {
            "B": {
            "notes": [
                "B <- A, C: A is the canonical activator of B; HIGH A should push B toward HIGH "
                "unless C is also HIGH, in which case C gates the signal and B should be ranked AVG "
                "or LOW regardless of A. LOW C releases that gate, making HIGH A sufficient for HIGH B. "
                "A common trap is treating A and C as independent additive inputs — C is a gating "
                "modulator, not a co-activator, so AVG A with LOW C should outrank AVG A with AVG C "
                "when downstream children of B signal a released state."
            ]
            }
        }
        }
        """
    # ── Output format ─────────────────────────────────────────────────────────
    footer = (
        "═══════════════════════════════════════════════════════\n"
        "OUTPUT FORMAT\n"
        "═══════════════════════════════════════════════════════\n"
        "Respond with a JSON object with a single key 'relationships'.\n"
        "Its value is a dict keyed by variable name (non-root variables only).\n"
        "Each entry must be an object with a single key 'notes' whose value is a "
        "LIST containing exactly ONE string (the full paragraph for that variable).\n\n"
        "Example skeleton (do not copy the placeholder text):\n"
        '{\n'
        '  "relationships": {\n'
        '    "VarA": { "notes": ["VarA <- Parent1, Parent2: …full paragraph…"] },\n'
        '    "VarB": { "notes": ["VarB <- Parent1: …full paragraph…"] }\n'
        '  }\n'
        '}\n'
    )

    return header + "\n".join(structure_lines) + "\n" + SYNTHETIC_EXAMPLE + "\n" + footer

# ── Helper ────────────────────────────────────────────────────────────────────
def _collect_siblings(bn: BayesianNetwork, var_name: str, parents: list[str]) -> list[str]:
    """Variables that share at least one parent with var_name (excluding itself)."""
    siblings = set()
    for parent in parents:
        for other_var, other_parents in bn.parents.items():
            if other_var != var_name and parent in other_parents:
                siblings.add(other_var)
    return sorted(siblings)

# ---- Gerador via LLM ----

def generate_relationship_notes_with_llm(
    bn: BayesianNetwork,
    llm_fn,
    output_path: Path | None = None,
) -> dict[str, list[str]]:
    prompt = build_relationship_notes_prompt(bn)
 
    print("Gerando relationship notes via LLM...")
    response = llm_fn(prompt, NetworkRelationshipResponse)
 
    result = {
        var: data.notes
        for var, data in response.relationships.items()
    }
 
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Relationship notes salvo em: {output_path}")
 
    return result


