# Online Phase — Running an MPE Query on the Semantic Maximizing Circuit

This document picks up where the offline phase ends. The student's slides eliminate the variables of the Arctic sea-ice network in the order $\pi = (X_5, X_4, X_3, X_2, X_1)$, and each elimination leaves behind one recorded table. The online phase runs a *query* on those tables in three steps: **set the indicators**, **propagate up**, **propagate down**. No LLM is ever called.

Throughout, we use the slide legend: **green** = value resolved by table lookup, **red** = value fixed by evidence.

## 0. What the offline phase left us

Five recorded tables, one per eliminated variable. Each row of a table records, for one configuration of the *context* variables, the maximizing configuration `pred(·)` chosen by the LLM at compilation time:

| table | context variables | recorded choice | rows |
|-------|-------------------|-----------------|------|
| $\tau_5$ | $X_3, X_4$ | pred($X_5$) | 9 |
| $\tau_4$ | $X_1, X_2, X_3$ | pred($X_4$) | 27 |
| $\tau_3$ | $X_1, X_2$ | pred($X_3$) | 9 |
| $\tau_2$ | $X_1$ | pred($X_2$) | 3 |
| $\tau_1$ | — (empty) | pred($X_1$) = **Low** | 1 |

The $\tau_5$ and $\tau_3$ tables are fully shown in the offline slides (both follow the same pattern: Low/Low → High, …, High/High → Low). We run two queries:

- **Query A (no evidence):** what is the most plausible configuration of the whole network?
- **Query B (evidence $X_5 =$ Low):** we observe low sea ice; what is the most plausible configuration of everything else?

## 1. Step 1 — Set the indicators

Every configuration of every variable has one indicator $\lambda$. The rule is the classical AC rule:

- If a variable is observed, its observed configuration gets $\lambda = 1$ and its other configurations get $\lambda = 0$.
- Unobserved variables get $\lambda = 1$ on **all** their configurations.

An indicator set to $0$ means: *every row of every table that mentions that configuration is deleted.*

- **Query A:** all indicators are $1$; nothing is deleted anywhere.
- **Query B:** $\lambda_{X_5=\text{Low}} = 1$, $\lambda_{X_5=\text{Medium}} = \lambda_{X_5=\text{High}} = 0$; everything else $1$. Since $X_5$ appears only in $\tau_5$, this targets the pred($X_5$) column of $\tau_5$.

## 2. Step 2 — Propagate up

In a numeric maximizing circuit, the upward pass evaluates the circuit leaves-to-root and pushes numbers through products and maxes. Here there are no numbers — but the pass still exists, and its job is the same: **carry the evidence upward**. The translation:

> The upward pass propagates **rows, not numbers**. We sweep the tables in elimination order ($\tau_5, \tau_4, \tau_3, \tau_2, \tau_1$) and delete every row that contradicts what we already know. What "flows up" from one table to the next is the set of its **surviving rows** — the configurations that can still explain the evidence. At the end, the narrowed tables tell us which parts of the network the evidence actually constrains.

Two deletion rules, applied at each table in the sweep:

1. **Evidence rule:** if the table's eliminated variable was observed, delete rows whose recorded `pred(·)` differs from the observation (this finishes the job the indicators started).
2. **Consistency rule:** delete rows that disagree with the surviving rows of the tables already swept, on the variables they share.

If every row of a table would be deleted, we keep the table as it was: the observation is atypical for that factor and simply does not propagate through it.

### Query A — nothing to do

All indicators are $1$ and there is no evidence, so no row contradicts anything: **every table survives in full**. The upward pass is a no-op, and all the work happens in the downward pass.

### Query B — evidence $X_5 =$ Low

**Sweep $\tau_5$ (evidence rule).** Delete every row whose pred($X_5$) is not Low:

| $X_3$ | $X_4$ | pred($X_5$) | |
|-------|-------|-------------|---|
| ~~Low~~ | ~~Low~~ | ~~High~~ | deleted |
| ~~Low~~ | ~~Medium~~ | ~~High~~ | deleted |
| ~~Low~~ | ~~High~~ | ~~Medium~~ | deleted |
| **Medium** | **High** | **Low** | survives |
| ~~Medium~~ | ~~Medium~~ | ~~Medium~~ | deleted |
| ~~Medium~~ | ~~Low~~ | ~~High~~ | deleted |
| **High** | **Medium** | **Low** | survives |
| **High** | **High** | **Low** | survives |
| ~~High~~ | ~~Low~~ | ~~Medium~~ | deleted |

9 rows → **3 survive**. Reading the survivors: *low sea ice can only be explained by the parent pairs $(X_3, X_4)$ = (Medium, High), (High, Medium), or (High, High).* That is the evidence, now expressed as a constraint on the middle layer.

**Sweep $\tau_4$ (consistency rule).** $\tau_4$ shares $X_3$ (context) and $X_4$ (its pred column) with $\tau_5$'s survivors. A row of $\tau_4$ survives iff its $(X_3, \text{pred}(X_4))$ pair is one of the three surviving pairs:

- rows with $X_3 =$ Low cannot survive (no surviving pair has $X_3 =$ Low);
- rows with $X_3 =$ High recorded pred($X_4$) = Low — the $(L,L,H)$ and $(H,H,H)$ rows appear in the offline slides — so they cannot survive either;
- rows with $X_3 =$ Medium survive exactly where pred($X_4$) = High.

Given the choices recorded at compilation, exactly **three rows survive** (an anti-diagonal, mirroring the pattern of $\tau_3$ and $\tau_5$):

| $X_1$ | $X_2$ | $X_3$ | pred($X_4$) |
|-------|-------|-------|-------------|
| Low | High | Medium | **High** |
| Medium | Medium | Medium | **High** |
| High | Low | Medium | **High** |

27 rows → 3 survive. Reading the survivors: *to explain $X_5 =$ Low through $\tau_4$, we need $X_3 =$ Medium, $X_4 =$ High, and the roots on the anti-diagonal $(X_1, X_2) \in \{$(Low, High), (Medium, Medium), (High, Low)$\}$.*

**Sweep $\tau_3$ (consistency rule).** The knowledge arriving from below is: $X_3 =$ Medium, and $(X_1, X_2)$ restricted to the anti-diagonal. Keep the rows whose pred($X_3$) = Medium **and** whose context is on the anti-diagonal:

| $X_1$ | $X_2$ | pred($X_3$) |
|-------|-------|-------------|
| **Low** | **High** | **Medium** |
| **Medium** | **Medium** | **Medium** |
| **High** | **Low** | **Medium** |

9 rows → 3 survive. The three survivors agree with the constraint from $\tau_4$ — the evidence is consistent so far.

**Sweep $\tau_2$ (consistency rule).** The surviving rows of $\tau_3$ still allow all three configurations of $X_2$ (High, Medium, Low all appear), so no row of $\tau_2$ is contradicted: 3 rows → 3 survive. *The evidence does not reach the roots' table for $X_2$.*

**Sweep $\tau_1$.** The root's table has a single row with an empty context — the a priori preference pred($X_1$) = Low. There is nothing to contradict: 1 row → 1 survives.

**Result of the upward pass:** the evidence narrowed $\tau_5$ (9→3), $\tau_4$ (27→3), and $\tau_3$ (9→3), but left $\tau_2$ and $\tau_1$ untouched. Three candidate paths survive; the downward pass will pick among them.

## 3. Step 3 — Propagate down

The downward pass reads off the MPE assignment, root to leaves, in reverse elimination order ($X_1, X_2, X_3, X_4, X_5$). In a numeric maximizing circuit this pass follows the max-branches from the root to the leaves; here every max was already resolved at compilation time, so the pass is pure lookup:

- If the variable was **observed**, assign the observed configuration (**red**).
- Otherwise, **look up the variable's own table** at the configuration of its context variables — all of which are already assigned, because the sweep goes in reverse elimination order (**green**). The lookup lands on a surviving row of the upward pass; if a variable was narrowed to a single configuration, the lookup confirms it.

### Query B — evidence $X_5 =$ Low

| step | variable | how | value |
|------|----------|-----|-------|
| 1 | $X_1$ | own table $\tau_1$, empty context: pred($X_1$) = Low | **Low** (green) |
| 2 | $X_2$ | own table $\tau_2$ at $X_1 =$ Low: pred($X_2$) = High | **High** (green) |
| 3 | $X_3$ | own table $\tau_3$ at $(X_1, X_2) =$ (Low, High): pred($X_3$) = Medium | **Medium** (green) |
| 4 | $X_4$ | own table $\tau_4$ at (Low, High, Medium): pred($X_4$) = High | **High** (green) |
| 5 | $X_5$ | evidence | **Low** (red) |

**MPE answer:** $(X_1, X_2, X_3, X_4, X_5) =$ **(Low, High, Medium, High, Low)**.

Sanity check: feed the hidden part back into $\tau_5$ — the surviving row ($X_3, X_4$) = (Medium, High) records pred($X_5$) = Low, exactly what was observed. The assignment does not just coexist with the evidence; the circuit's own tables *predict* it.

### Query A — no evidence

All tables are unnarrowed, so the downward pass is five plain lookups: $\tau_1$ gives Low, $\tau_2$ at Low gives High, $\tau_3$ at (Low, High) gives Medium, $\tau_4$ at (Low, High, Medium) gives High, and — the only difference — $X_5$ is green, not red: $\tau_5$ at (Medium, High) gives pred($X_5$) = Low.

**MPE answer:** **(Low, High, Medium, High, Low)** — with $X_5$ resolved by lookup rather than fixed by evidence.

## 4. Closing note

Both queries return the same assignment, and that is worth a slide of its own: the observation $X_5 =$ Low *confirms* the network's default most-plausible explanation. The machinery differs, though — Query A is decided entirely by downward lookups, while Query B first narrows the middle layer of the circuit (27 rows of $\tau_4$ down to 3, and 9 rows of $\tau_3$ and $\tau_5$ down to 3 each) so that the evidence provably flows through every table that can explain it. And because the tables were compiled with no evidence assumed, the same circuit answers any other query — other observations, partial observations, or none — with nothing but indicator settings, row deletions, and lookups.
