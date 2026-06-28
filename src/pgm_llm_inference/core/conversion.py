from os import name
from pgm_llm_inference.models import BayesianNetwork, Variable, Factor
from pathlib import Path
from pgmpy.models import DiscreteBayesianNetwork

def convert_pgmpy_model(model) -> BayesianNetwork:
    """
    Convert a pgmpy BayesianModel into the library's custom BayesianNetwork format.

    The function:
        - Creates Variable objects for each pgmpy node
        - Reconstructs all CPDs as Factor objects
        - Preserves variable order, domains, and CPT values

    Args:
        model: A pgmpy BayesianModel already loaded with CPDs.

    Returns:
        A BayesianNetwork instance equivalent to the pgmpy model.

    Example:
        >>> model = BIFReader("alarm.bif").get_model()
        >>> network = convert_pgmpy_model(model)
    """
    import numpy as np

    network = BayesianNetwork()
    var_map: dict[str, Variable] = {}

    # ---- variables ----
    for var_name in model.nodes():
        cpd = model.get_cpds(var_name)
        domain = list(cpd.state_names[var_name])
        variable = Variable(name=var_name, states=domain)
        var_map[var_name] = variable
        network.add_variable(variable)

    # ---- factors ----
    for cpd in model.get_cpds():
        variable = var_map[cpd.variable]
        parents = [var_map[p] for p in (cpd.get_evidence() or [])]

        scope_vars = [variable] + parents

        cpd_var_names = list(cpd.variables)
        scope_var_names = [v.name for v in scope_vars]

        axis_permutation = [
            cpd_var_names.index(var_name)
            for var_name in scope_var_names
        ]

        values = np.transpose(np.array(cpd.values), axes=axis_permutation)

        factor = Factor(scope=scope_vars, values=values)
        network.add_factor(factor)

    return network

def parse_dne_to_pgmpy(path: str):
    import re
    import numpy as np
    from pathlib import Path
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD

    text = Path(path).read_text(encoding="ISO-8859-1")

    # --------------------------------------------------
    # Remove comentários //
    # --------------------------------------------------
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)

    # --------------------------------------------------
    # Regex helpers
    # --------------------------------------------------
    NODE_RE = re.compile(
        r'node\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{(.*?)\};',
        re.DOTALL
    )

    NUMBER_RE = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'

    nodes: dict[str, dict] = {}

    # --------------------------------------------------
    # Parse nodes
    # --------------------------------------------------
    for name, block in NODE_RE.findall(text):

        # ---------- STATES or LEVELS ----------
        states = None

        states_m = re.search(r'states\s*=\s*\((.*?)\);', block, re.DOTALL)
        if states_m:
            raw = states_m.group(1)
            pairs = re.findall(r'"([^"]+)"|([A-Za-z0-9_]+)', raw)
            states = [a or b for a, b in pairs]
        else:
            levels_m = re.search(r'levels\s*=\s*\((.*?)\);', block, re.DOTALL)
            if levels_m:
                nums = list(map(float, re.findall(NUMBER_RE, levels_m.group(1))))
                # Netica: N levels -> N-1 intervals
                states = [
                    f"{nums[i]}-{nums[i+1]}"
                    for i in range(len(nums) - 1)
                ]

        # Se não tem states nem levels, ignora (visual, utility, etc.)
        if states is None:
            continue

        # ---------- PARENTS ----------
        parents_m = re.search(r'parents\s*=\s*\((.*?)\);', block, re.DOTALL)
        if parents_m:
            raw = parents_m.group(1).strip()
            parents = [] if raw == "" else re.findall(
                r'[A-Za-z_][A-Za-z0-9_]*', raw
            )
        else:
            parents = []

        # ---------- PROBS or BELIEF ----------
        probs_m = re.search(r'probs\s*=\s*\((.*?)\);', block, re.DOTALL)
        belief_m = re.search(r'belief\s*=\s*\((.*?)\);', block, re.DOTALL)

        matrix = None

        if probs_m:
            numbers = list(map(float, re.findall(NUMBER_RE, probs_m.group(1))))
            card = len(states)

            if len(numbers) % card != 0:
                raise ValueError(
                    f"Invalid CPD size for node '{name}': "
                    f"{len(numbers)} values for {card} states"
                )

            num_cols = len(numbers) // card

            # Netica lists rows by parent configuration
            matrix = [
                numbers[i * card:(i + 1) * card]
                for i in range(num_cols)
            ]

        elif belief_m:
            numbers = list(map(float, re.findall(NUMBER_RE, belief_m.group(1))))
            if len(numbers) != len(states):
                raise ValueError(
                    f"Invalid belief size for node '{name}'"
                )
            matrix = [numbers]

        else:
            raise ValueError(f"No probabilities found for node '{name}'")

        nodes[name] = {
            "states": states,
            "parents": parents,
            "cpd": matrix,
        }

    if not nodes:
        raise ValueError("No nodes parsed from .dne file")

    # --------------------------------------------------
    # Build pgmpy model
    # --------------------------------------------------
    model = DiscreteBayesianNetwork()

    for name in nodes:
        model.add_node(name)

    for name, info in nodes.items():
        for p in info["parents"]:
            if p in nodes:
                model.add_edge(p, name)

    # --------------------------------------------------
    # Build CPDs
    # --------------------------------------------------
    for name, info in nodes.items():
        states = info["states"]
        parents = [p for p in info["parents"] if p in nodes]
        matrix = info["cpd"]

        variable_card = len(states)

        expected_cols = 1
        for p in parents:
            expected_cols *= len(nodes[p]["states"])

        values = np.array(matrix, dtype=float).T

        if values.shape != (variable_card, expected_cols):
            raise ValueError(
                f"CPD shape mismatch for node '{name}': "
                f"expected {(variable_card, expected_cols)}, got {values.shape}"
            )

        state_names = {name: states}
        for p in parents:
            state_names[p] = nodes[p]["states"]

        cpd = TabularCPD(
            variable=name,
            variable_card=variable_card,
            values=values,
            evidence=parents or None,
            evidence_card=[len(nodes[p]["states"]) for p in parents] if parents else None,
            state_names=state_names,
        )

        model.add_cpds(cpd)

    assert model.check_model()
    return model
