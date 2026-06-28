# PGM-LLM Inference

Biblioteca em Python para inferência em Redes Bayesianas com estratégias de
eliminação plugáveis — incluindo uma abordagem de inferência MAP/MPE guiada
por LLM (raciocínio semântico em vez de eliminação numérica).

## Visão geral

O núcleo implementa Variable Elimination (VE) com três estratégias
intercambiáveis (Strategy Pattern):

1. **Sum-Product** — inferência posterior tradicional via marginalização.
2. **Max-Product** — MAP/MPE exato via otimização numérica.
3. **LLM Semantic** — MAP/MPE via raciocínio do LLM sobre o significado das
   variáveis, em vez de operações numéricas em CPTs.

Em cima desse núcleo existe uma camada de **experimentos** (`experiment/`)
que monta datasets reais (`.bif`, `.xdsl`, `.dne`), gera metadados/contexto
via LLM quando necessário, roda os três métodos lado a lado e compara os
resultados.

## Instalação

```bash
uv sync                 # dependências de runtime
uv sync --extra dev      # + pytest, ruff
```

Copie `.env.example` para `.env` e ajuste as variáveis (chave de API OpenAI,
modelo, timeouts, etc.) antes de rodar qualquer fluxo que use LLM real.

## Quick start (uso direto da biblioteca)

```python
import numpy as np
from pgm_llm_inference import (
    Variable, Factor, BayesianNetwork,
    SumProductStrategy, InferenceEngine,
)

rain = Variable(name="Rain", domain=["yes", "no"])
grass = Variable(name="GrassWet", domain=["yes", "no"])

network = BayesianNetwork()
network.add_variable(rain)
network.add_variable(grass)

p_rain = Factor(scope=[rain], values=np.array([0.2, 0.8]))
network.add_factor(p_rain)

engine = InferenceEngine(network=network, strategy=SumProductStrategy())
result = engine.query(query_vars=["Rain"], evidence={"GrassWet": "yes"})

print(result["result_factor"].values)  # P(Rain | GrassWet=yes)
```

### Usando um LLM real (estratégia LLM Semantic)

```python
from pgm_llm_inference import LLMSemanticStrategy, InferenceConfig
from pgm_llm_inference.strategies.llm.openai import create_openai_llm_function

config = InferenceConfig()  # lê variáveis de .env (prefixo PGM_)
llm_fn = create_openai_llm_function(config)
strategy = LLMSemanticStrategy(llm_query_fn=llm_fn)
```

Para um LLM local (servidor compatível com OpenAI, ex. LM Studio/Ollama),
use `pgm_llm_inference.strategies.llm.local.local_llm_structured` no lugar,
ou `experiment.llm_factory.build_llm_fn(use_real_llm=False, use_local_llm=True)`
para deixar a escolha configurável.

## Estrutura do projeto

```
src/pgm_llm_inference/
├── models/             Variable, Factor, BayesianNetwork (Pydantic)
├── core/               config, ordenação de eliminação, operações de fatores
├── inference/          motor de Variable Elimination (engine, ve_algorithm)
├── strategies/         Sum-Product, Max-Product e LLM (base.py define a interface)
│   └── llm/
│       ├── llm_mpe_sachs.py   tipos + pipeline LLM-MPE (parse de .bif, prompts,
│       │                      bucket elimination semântica, reconstrução, audit)
│       ├── semantic.py        LLMSemanticStrategy (MAP por variável)
│       ├── openai.py          cliente LLM via API OpenAI
│       ├── local.py           cliente LLM via servidor local (streaming)
│       └── prompts.py         templates de prompt (simple / few-shot / CoT)
├── experiment/         orquestração de experimentos
│   ├── compiled_inference.py   compile (1x por dataset) + infer (por evidência)
│   ├── experiment.py            run_single_map_experiment / run_single_mpe_experiment
│   ├── metadata_generation.py   geração de metadados e relationship-notes via LLM
│   ├── runners.py / batch.py / sampling.py / llm_factory.py / metadata.py
├── io/                 carregamento de redes (.bif/.bif.gz/.xdsl/.dne)
├── evaluation/         métricas (acerto LLM vs. MPE exato)
├── logging/            registro de resultados em JSONL/CSV
├── utils.py            parsing de saída do LLM (extração de JSON/texto)
└── scripts/
    ├── run/            scripts para EXECUTAR experimentos
    │   ├── main.py             batch completo: múltiplos datasets e evidências
    │   └── main_single_run.py  exemplo mínimo: 1 dataset, 1 evidência
    ├── analysis/       scripts para ANALISAR resultados já gerados
    │   ├── metrics_3.py        gráficos e métricas (rodar manualmente após os experimentos)
    │   ├── reinference.py      segunda passada de verificação via Markov Blanket
    │   └── get_table.py        estatísticas estruturais dos datasets (Markov Blanket médio, etc.)
    └── data_prep/      utilitários de preparação de dataset
        └── change_nodes_name.py
```

Pastas vazias por padrão (geradas/baixadas em tempo de execução, fora do
controle de versão — ver `.gitignore`): `datasets/`, `metadata/`,
`relationships/`, `logs/`.

## Rodando experimentos

```bash
# Batch completo (múltiplos datasets/evidências)
uv run python -m pgm_llm_inference.scripts.run.main

# Um único dataset/evidência (exemplo mínimo)
uv run python -m pgm_llm_inference.scripts.run.main_single_run

# Análises pós-hoc (depois de já ter rodado experimentos e gerado logs)
uv run python -m pgm_llm_inference.scripts.analysis.metrics_3
uv run python -m pgm_llm_inference.scripts.analysis.reinference
uv run python -m pgm_llm_inference.scripts.analysis.get_table
```

## Configuração

Todas as opções de runtime vêm de `InferenceConfig` (`core/config.py`),
populada a partir de variáveis de ambiente com prefixo `PGM_` (ver
`.env.example`): nível de log, epsilon numérico, heurística de ordenação
de eliminação, timeout/retries do LLM, e credenciais da API OpenAI.

## Desenvolvimento

```bash
uv run pytest
uv run ruff check src/
uv run ruff format src/
```

## Arquitetura

A biblioteca usa Strategy Pattern para alternar entre métodos de inferência:

- **models/** — tipos de dados imutáveis (Pydantic) para variáveis, fatores
  e a rede.
- **core/factor_ops.py** — operações numéricas (einsum, marginalização,
  maximização) compartilhadas pelas estratégias numéricas.
- **strategies/** — cada estratégia implementa a interface `EliminationStrategy`
  (`strategies/base.py`); plugável no `InferenceEngine` sem alterar o motor.
- **inference/engine.py** — orquestra o algoritmo de Variable Elimination
  usando a estratégia injetada.
- **experiment/compiled_inference.py** — pipeline de produção da abordagem
  LLM-MPE: compila as mensagens semânticas uma vez por dataset (custo: N
  chamadas LLM) e reutiliza essa compilação para inferir múltiplas
  evidências (custo: ~2 chamadas LLM por evidência).
