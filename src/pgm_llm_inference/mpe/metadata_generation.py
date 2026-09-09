"""
metadata_generation.py
=======================
Geração via LLM de metadados qualitativos por variável (display_name,
expert_note e state_meanings), persistidos uma vez por dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .types import VariableMetadata
from ..models import BayesianNetwork

# ---- Modelos de saída ----

class NetworkMetadataResponse(BaseModel):
    metadata: dict[str, VariableMetadata]

# ---- Prompt builder ----

def build_metadata_generation_prompt(bn: BayesianNetwork) -> str:
    """
    Build a prompt that instructs an LLM to generate structured metadata
    for every variable in the given Bayesian Network.
 
    The returned prompt is a self-contained instruction string suitable for
    use as the sole user message to a capable chat-completion model.  The
    expected model output is a single JSON object — one key per variable —
    matching the schema described inside the prompt.
 
    Args:
        bn: A BayesianNetwork instance whose `variables`, `parents`, and
            `children_map` properties are fully populated.
 
    Returns:
        A string prompt ready to be sent to an LLM.
    """
    parents_map: dict[str, tuple[str, ...]] = bn.parents
    children_map: dict[str, tuple[str, ...]] = bn.children_map()
 
    # ------------------------------------------------------------------ #
    # Build a structured variable table to embed in the prompt             #
    # ------------------------------------------------------------------ #
    var_lines: list[str] = []
    for name, var in bn.variables.items():
        var_parents = parents_map.get(name, ())
        var_children = children_map.get(name, ())
 
        if not var_parents and var_children:
            role = "root (no parents; exogenous driver)"
        elif var_parents and not var_children:
            role = "leaf (no children; terminal readout)"
        elif var_parents and var_children:
            role = "intermediate (has both parents and children)"
        else:
            role = "isolated (no edges)"
 
        domain_str = ", ".join(f'"{s}"' for s in var.states)
        parents_str = ", ".join(var_parents) if var_parents else "—"
        children_str = ", ".join(var_children) if var_children else "—"
 
        var_lines.append(
            f"  {name}\n"
            f"    domain   : [{domain_str}]\n"
            f"    role     : {role}\n"
            f"    parents  : {parents_str}\n"
            f"    children : {children_str}"
        )
 
    variables_section = "\n\n".join(var_lines)
    n = len(bn.variables)
 
    # ------------------------------------------------------------------ #
    # Compose the prompt                                                   #
    # ------------------------------------------------------------------ #
    prompt = f"""\
You are an expert knowledge engineer specialising in probabilistic graphical \
models and domain knowledge elicitation.
 
Your task is to generate structured metadata for every variable in the \
Bayesian Network described below.  This metadata will be used as a \
knowledge-enriched context layer (analogous to a RAG retrieval system) to \
guide a language model that performs Most Probable Explanation (MPE) \
inference without access to the network's conditional probability tables \
(CPTs).  The inference model will rely solely on semantics and world \
knowledge, anchored by the metadata you produce.
 
═══════════════════════════════════════════════════════════════════════════
WHY THIS METADATA MATTERS
═══════════════════════════════════════════════════════════════════════════
 
When performing MPE inference without CPTs, a language model is prone to \
several recurring failure modes:
 
• Causal reversal — the model infers a parent's state from its children \
  rather than the other way around, or propagates evidence in the wrong \
  direction.
• Passive relay error — the model copies a parent's state directly to all \
  children without accounting for attenuation, amplification, or \
  domain-specific modulation.
• Sibling conflation — the model treats two children of the same parent as \
  independent evidence and combines them (e.g., averages them), when they \
  are parallel consequences of the same upstream cause.
• Base-rate neglect — the model assigns a rare state because it is \
  semantically salient, ignoring that the common state is overwhelmingly \
  more probable by default.
• Semantic blurring — the model treats two distinct variables as \
  interchangeable because they sound similar or share surface correlates.
 
Your metadata must inoculate the inference model against these failure \
modes by being explicit, actionable, and tightly grounded in the actual \
network structure.
 
═══════════════════════════════════════════════════════════════════════════
BAYESIAN NETWORK STRUCTURE  ({n} variables)
═══════════════════════════════════════════════════════════════════════════

Network name: {bn.name or "unnamed"} 

{variables_section}
 
═══════════════════════════════════════════════════════════════════════════
OUTPUT SPECIFICATION
═══════════════════════════════════════════════════════════════════════════
 
Return a SINGLE JSON OBJECT.  No preamble.  No markdown fences.  No \
trailing commentary.  The top-level object must have exactly one key, \
"metadata", whose value is a dict with one entry per variable:
 
{{
  "metadata": {{
    "<VARIABLE_NAME>": {{
      "display_name": "<string>",
      "description": "<string>",
      "expert_note": "<string>",
      "aliases": ["<string>", ...],
      "state_meanings": {{
        "<EXACT_STATE_STRING>": "<string>",
        ...
      }}
    }},
    ...
  }}
}}
 
═══════════════════════════════════════════════════════════════════════════
FIELD-BY-FIELD INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════
 
── display_name ──────────────────────────────────────────────────────────
 
A short, human-readable label (2–6 words).  Expand abbreviations where \
helpful.  Capture the real-world concept, not just the raw identifier.
 
── description ───────────────────────────────────────────────────────────
 
1–3 sentences.  State factually:
  1. What this variable represents in the domain.
  2. Its topological role (root / intermediate / leaf / isolated).
  3. Its direct parents (the upstream drivers) and children (the downstream \
readouts) referred to by their exact variable names.
 
Do not editorialize.  Do not repeat the expert_note content here.
 
── expert_note ───────────────────────────────────────────────────────────
 
4–8 sentences.  This is the most important field for inference quality.  \
Address ALL of the following sub-points that apply to this variable's role:
 
  1. SEMANTIC CORE — What does this variable fundamentally measure or \
represent?  How is it distinct from superficially similar concepts or \
variables in this network?
 
  2. PARENT INFLUENCE — For each parent: in which direction and by what \
mechanism does the parent affect this variable (enabling, suppressing, \
amplifying, gating, modulating)?  State the causal logic plainly.  \
*Root nodes skip this sub-point.*
 
  3. CHILD COHERENCE — What constraints do the children impose as \
coherence checks?  If this variable is set to a particular state, which \
child states become more or less plausible, and how should the inference \
model use those as back-pressure?  *Leaf and isolated nodes skip this \
sub-point.*
 
  4. SIBLING DYNAMICS — If this variable shares a parent with one or more \
sibling variables, explain how the siblings relate (e.g., parallel \
reporters of the same cause, complementary dimensions, different lags) \
and how the inference model should use them as mutual cross-checks \
WITHOUT treating them as independent evidence.  *Skip if there are no \
siblings.*
 
  5. REASONING TRAPS — Name and describe at least one concrete, specific \
wrong inference the language model is likely to make when this variable \
is assigned a particular state.  Be explicit: describe what the wrong \
inference is, why it is tempting, and what the correct reasoning is \
instead.  Vague cautions ("be careful") are not acceptable here.
 
Role-specific focus:
  • Root nodes: emphasise (1), (3), (4), (5) — no parent influence to \
describe.
  • Leaf nodes: emphasise (1), (2), (4), (5) — no children to \
constrain.
  • Isolated nodes: (1) and (5) only.
  • Intermediate nodes: all five sub-points are expected.

  
 
── aliases ───────────────────────────────────────────────────────────────
 
2–4 alternative names, abbreviations, or synonyms that a natural-language \
description of the domain might use for this concept.  These help the \
downstream inference model recognise the variable under different \
phrasings.
 
── state_meanings ────────────────────────────────────────────────────────
 
One entry per value in the variable's domain.  Keys must be the EXACT \
state strings listed in the domain above (copy them verbatim).  Each \
value is 1–3 sentences covering:
 
  • What this state means in real-world terms.
  • Which parent configuration is most consistent with this state \
(skip for root / isolated nodes).
  • What downstream states this implies for the children \
(skip for leaf / isolated nodes).
 
Do not simply rephrase the state name.  Ground every meaning in the \
causal context.
 
═══════════════════════════════════════════════════════════════════════════
HARD CONSTRAINTS
═══════════════════════════════════════════════════════════════════════════
 
1. The top-level JSON object must have exactly one key: "metadata".  All \
   variable entries go inside that nested dict.  Do not hoist variables to \
   the top level.
2. Generate metadata for ALL {n} variables listed above inside "metadata".  \
   Skipping any variable is an error.
3. Use the EXACT variable names from the table as keys inside "metadata".  \
   Do not rename, abbreviate, or reorder them.
4. Use the EXACT state strings from each variable's domain as keys inside \
   state_meanings.  Do not rename, merge, add, or drop states.
5. All causal claims must be consistent with the network structure above.  \
   Do not invent parents, children, or edges that are not listed.
6. The expert_note MUST name at least one concrete, specific reasoning \
   trap (sub-point 5).  A generic caution without a named wrong inference \
   does not satisfy this requirement.
7. Output only the JSON object.  No explanation, no markdown, no \
   surrounding text of any kind.
"""
    return prompt

# ---- Função principal ----
 
def generate_metadata_with_llm(
    bn,
    llm_fn,
    output_path: Path | None = None,
) -> dict:
    
    prompt = build_metadata_generation_prompt(bn)
    response = llm_fn(prompt, NetworkMetadataResponse)
 
    metadata_dict = {
        var: meta.model_dump()
        for var, meta in response.metadata.items()
    }
 
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metadata_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Metadata salvo em: {output_path}")
 
    return metadata_dict
