"""
reinference.py
===============
Post-processing de logs de experimentos já rodados: para cada variável
prevista pelo LLM, verifica consistência com seu Markov Blanket e corrige
predições que destoam sistematicamente do que o blanket sugere (ex.: viés
para valores "neutros"/centrais do domínio).

Lê um arquivo de log .jsonl (gerado por logging/experiment_logger.py),
roda uma segunda passada de verificação via LLM, e escreve um novo log
"<nome>_re.jsonl" com as predições corrigidas.

Uso:
    python -m pgm_llm_inference.scripts.reinference
"""

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from pgm_llm_inference.models import BayesianNetwork
from pgm_llm_inference.core.config import InferenceConfig
from pgm_llm_inference.io.loaders import load_network
from pgm_llm_inference.experiment.llm_factory import build_llm_fn

config = InferenceConfig()

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"


class ReInferenceResult(BaseModel):
    reasoning: str = Field(description="MPE analysis explanation.")
    predictions: dict[str, str] = Field(description="The re-inferred values for the target variables.")


def _run_llm_fn(llm_fn, prompt: str) -> dict[str, str]:
    """Chama llm_fn e normaliza o retorno para um dict {variavel: valor}."""
    result = llm_fn(prompt, ReInferenceResult)
    if isinstance(result, ReInferenceResult):
        return result.predictions
    if isinstance(result, dict):
        return result.get("predictions", result)
    raise TypeError(f"Unexpected LLM result type: {type(result)}")


def _print_diff_table(target_vars: list, old_preds: dict, new_preds: dict, map_assignment: dict):
    col_w = 18
    header = f"{'Variable':<{col_w}} {'Before':>{col_w}} {'After':>{col_w}} {'GT':>{col_w}} {'Changed?':>{10}}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for var in target_vars:
        old_v, new_v, gt_v = old_preds.get(var), new_preds.get(var), map_assignment.get(var, "?")
        changed = "YES" if old_v != new_v else "no"
        print(f"{var:<{col_w}} {str(old_v):>{col_w}} {str(new_v):>{col_w}} {str(gt_v):>{col_w}} {changed:>{10}}")


def run_reinference(
    log_file_path: str,
    use_local: bool = False,
    datasets_to_run: Optional[List[str]] = None,
    datasets_dir: Path = DATASETS_DIR,
):
    llm_fn = build_llm_fn(use_real_llm=not use_local, use_local_llm=use_local)
    dataset_cache = {}

    input_path = Path(log_file_path)
    output_path = input_path.with_name(f"{input_path.stem}_re{input_path.suffix}")

    with open(input_path, 'r', encoding='utf-8') as f:
        all_lines = [json.loads(l) for l in f if l.strip()]

    if datasets_to_run:
        filtered_data = [d for d in all_lines if d.get("dataset") in datasets_to_run]
    else:
        filtered_data = all_lines

    total_to_process = len(filtered_data)
    
    print("=" * 60)
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Total in original: {len(all_lines)}")
    print(f"  Total to process (filtered): {total_to_process}")
    print("=" * 60)

    error_count = 0

    with open(output_path, 'a', encoding='utf-8') as f_out:
        for i, data in enumerate(filtered_data, 1):
            print(f"\n[{i:>4}/{total_to_process}] Processing Dataset: {data.get('dataset')}", flush=True)

            try:
                dataset_name = data.get("dataset")
                if dataset_name not in dataset_cache:
                    dataset_path = datasets_dir / dataset_name
                    network, _ = load_network(str(dataset_path), None, llm_fn)
                    node_domains = {name: var.domain for name, var in network.variables.items()}
                    edges, markov_blankets = _extract_edges_and_markov_blankets(network)
                    dataset_cache[dataset_name] = (network, node_domains, edges, markov_blankets)
                else:
                    network, node_domains, edges, markov_blankets = dataset_cache[dataset_name]

                evidences = data.get("evidence", {})
                predictions = data.get("llm_predictions", {})
                map_assign = data.get("map_assignment", {})
                target_vars = [k for k in predictions if k not in evidences]

                if not target_vars:
                    f_out.write(json.dumps(data) + '\n')
                    continue

                domain_info = {var: node_domains.get(var, "Unknown") for var in target_vars}

                # Only include edges relevant to target variables
                nodes_of_interest = set(target_vars) | set(evidences.keys())
                relevant_edges = [
                    e for e in edges
                    if e[0] in nodes_of_interest or e[1] in nodes_of_interest
                ]

                # Only include Markov blankets for target variables
                relevant_blankets = {
                    var: markov_blankets[var]
                    for var in target_vars
                    if var in markov_blankets
                }

                prompt = f"""
### TASK: PROBABILISTIC JOINT CONSISTENCY VERIFICATION

You are a probabilistic consistency checker for Bayesian network inference.
A first-pass predictor assigned values to target variables independently, which may
have introduced systematic biases — in particular, a tendency to predict neutral or
middle-of-the-domain values even when blanket evidence suggests otherwise.

Your role: identify and correct variables whose current prediction is inconsistent
with their Markov blanket. Scrutinize neutral and middle values as carefully as
extreme ones — they are not inherently safer or more probable.

---
### FIXED EVIDENCE (immutable — do NOT modify or output these)
{json.dumps(evidences, indent=2)}

### TARGET VARIABLES AND THEIR DOMAINS
Each domain is a finite set of mutually exclusive categories. No category is
a priori more probable than another — do not treat middle or neutral-sounding
values as default-correct.
{json.dumps(domain_info, indent=2)}

### MARKOV BLANKETS
Each variable is conditionally independent of all other nodes given its Markov
blanket (parents, children, co-parents). Reason about each variable ONLY via
its blanket nodes — not global network themes.
{json.dumps(relevant_blankets, indent=2)}

### CURRENT JOINT ASSIGNMENT
{json.dumps({k: predictions[k] for k in target_vars}, indent=2)}

---
### VERIFICATION PROTOCOL

**STEP 1 — Blanket Snapshot**
List each blanket node and its current value (from evidence + predictions).

**STEP 2 — Directional Signals**
For each blanket node, assess whether it pulls this variable toward a specific
domain value or away from its current prediction. Classify each signal as:
  - "pulls toward <value>" / "pulls away from current" / "neutral"

**STEP 3 — Signal Aggregation**
Count signals in each direction. If the majority of non-neutral blanket signals
point away from the current prediction, correction is warranted.

**STEP 4 — Value Selection**
Choose the domain value most supported by the directional signals.
Do not default to the middle of the domain — select whatever value the
signals most consistently support, regardless of its position in the domain.

---
### REASONING FORMAT

For each target variable write exactly:

[<VarName>]
  Blanket values: <node: value, ...>
  Current prediction: <value>
  Directional signals: <node → signal, ...>
  Net signal: <toward current / away from current / mixed>
  Decision: PRESERVE / CORRECT
  If CORRECT → new value: <value> | Reasoning: <which signals drive this choice>

---
### BIASES TO COUNTERACT

- Do not anchor on the semantic meaning of variable names or domain value labels.
- Do not treat middle or neutral domain values as default-safe — they can be
  wrong just as often as values at either extreme.
- A variable whose blanket nodes consistently suggest a direction away from its
  current value is a correction candidate regardless of what that value is.
- Do not let a single extreme evidence node justify sweeping corrections across
  unrelated variables.

---
### OUTPUT

After reasoning, return ONLY a flat JSON with ALL target variables:

{{
    "<var1>": "<value>",
    ...
    "<varN>": "<value>"
}}

Values must match the domain list exactly. No extra fields.
"""

                success = False
                final_corrected = {}
                for attempt in range(3):
                    try:
                        result = _run_llm_fn(llm_fn, prompt)
                        if all(v in result for v in target_vars):
                            final_corrected = result
                            success = True
                            break
                    except Exception:
                        continue

                if not success:
                    error_count += 1
                    final_corrected = {v: predictions.get(v) for v in target_vars}

                _print_diff_table(target_vars, predictions, final_corrected, map_assign)

                new_entry = data.copy()
                new_entry["llm_predictions_original"] = {k: predictions[k] for k in target_vars}
                updated_preds = dict(predictions)
                updated_preds.update(final_corrected)

                new_hits = sum(1 for v, val in updated_preds.items() if v in map_assign and val == map_assign[v])
                new_eval = len([v for v in updated_preds if v in map_assign])

                new_entry.update({
                    "hits": new_hits,
                    "evaluated_length": new_eval,
                    "log_accuracy": new_hits / new_eval if new_eval > 0 else 0,
                    "llm_predictions": updated_preds
                })

                f_out.write(json.dumps(new_entry) + '\n')
                f_out.flush()

            except Exception as e:
                error_count += 1
                print(f"  [ERROR] {e}")

    print("\n" + "=" * 60)
    print(f"  DONE. Processed {total_to_process} lines.")
    print("=" * 60)


def _extract_edges_and_markov_blankets(network: BayesianNetwork) -> tuple[list, dict]:
    """
    Derives edges and Markov blankets from the network's factors.
    In each factor, scope[0] is the child and scope[1:] are its parents.
    """
    parents: dict[str, set] = {name: set() for name in network.variables}
    children: dict[str, set] = {name: set() for name in network.variables}

    edges = []
    for factor in network.factors:
        if len(factor.scope) < 2:
            continue  # root node (no parents), no edges to add
        child = factor.scope[0].name
        for parent_var in factor.scope[1:]:
            parent = parent_var.name
            edges.append((parent, child))
            parents[child].add(parent)
            children[parent].add(child)

    blankets = {}
    for node in network.variables:
        node_parents = parents.get(node, set())
        node_children = children.get(node, set())
        coparents = set()
        for child in node_children:
            coparents.update(parents.get(child, set()))
        coparents.discard(node)

        blankets[node] = {
            "parents": sorted(node_parents),
            "children": sorted(node_children),
            "co_parents": sorted(coparents),
        }

    return edges, blankets

if __name__ == "__main__":
    LOG_PATH = config.log_file_name 

    datasets = [
        # "cryptocurrency.xdsl",
        # "bankruptcy.bif",
        # "coral1.bif",
        # "Coronary_Risk.dne",
        # "crimescene.bif",
        "sachs.bif",
        # "gonorrhoeae.bif",
        # "insurance.bif",
        # "child.bif.gz",
    ]
    run_reinference(LOG_PATH, use_local=False, datasets_to_run=datasets)