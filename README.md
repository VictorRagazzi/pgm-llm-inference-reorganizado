# PGM-LLM Inference

Projeto de pesquisa para comparar inferência exata em Redes Bayesianas com
uma abordagem semântica baseada em Large Language Models.

O projeto possui dois mecanismos distintos:

- inferência numérica por Variable Elimination, com Sum-Product e Max-Product;
- inferência LLM-MPE, que compila decisões semânticas uma vez e reutiliza essas
  decisões para diferentes configurações de evidência.

O entry point do experimento principal é
`pgm_llm_inference.scripts.run.main`.

## Fluxo principal

```text
scripts/run/main.py
        │
        ├── io.loaders.load_network
        │       carrega e converte a rede para o modelo interno
        │
        ├── mpe.cache.load_or_compile
        │       ├── reutiliza uma compilação válida; ou
        │       └── mpe.compile.compile_semantic_messages
        │               gera metadados, notas e mensagens semânticas
        │
        ├── experiment.batch.run_batch
        │       produz configurações de evidência e consulta
        │
        └── experiment.runner.run_experiment
                ├── Max-Product exato, usado como referência
                ├── inferência semântica a partir da compilação
                ├── cálculo das métricas
                └── persistência do resultado em JSONL
```

A compilação é a etapa que pode chamar o LLM. Depois que ela está salva, a
inferência em `mpe.infer` usa apenas lookup e reconstrução determinística por
backpointers, sem novas chamadas ao modelo.

## Organização

```text
src/pgm_llm_inference/
├── models/                 modelos canônicos da rede
│   ├── variable.py         variável discreta e seus estados
│   ├── factor.py           tabela multidimensional de probabilidades
│   └── bayesian_network.py variáveis, fatores e topologia
│
├── core/                   operações fundamentais
│   ├── config.py           configuração de runtime e ambiente
│   ├── conversion.py       conversão dos formatos externos
│   ├── factor_ops.py       produto, redução, soma e maximização de fatores
│   └── ordering.py         heurísticas de ordem de eliminação
│
├── inference/              motor numérico de Variable Elimination
│   ├── engine.py           validação e orquestração da consulta
│   ├── ve_algorithm.py     algoritmo de eliminação
│   └── postprocessing.py   normalização e reconstrução numérica
│
├── strategies/             operações específicas de cada estratégia
│   ├── sum_product.py       inferência posterior
│   ├── max_product.py       MAP/MPE exato
│   └── llm/                 clientes e parsing de respostas LLM
│
├── mpe/                    pipeline semântico LLM-MPE
│   ├── compile.py           compilação das mensagens semânticas
│   ├── cache.py             persistência e versionamento da compilação
│   ├── infer.py             inferência sem novas chamadas LLM
│   ├── reconstruction.py    reconstrução determinística por backpointers
│   ├── bucket.py            construção e validação dos buckets
│   ├── prompt_builders.py   prompts usados durante a compilação
│   ├── metadata_generation.py
│   ├── relationship_generation.py
│   ├── graph.py             operações de grafo
│   ├── state_semantics.py   interpretação dos estados
│   ├── client.py            cliente estruturado do LLM
│   ├── io.py                parsing BIF e normalização semântica
│   └── types.py             schemas do pipeline
│
├── experiment/             execução dos experimentos
│   ├── config.py            ExperimentConfig e modos MAP/MPE
│   ├── batch.py             geração do batch
│   ├── sampling.py          estratégias de amostragem de evidência
│   ├── runner.py            comparação exata versus semântica
│   ├── experiment.py        operações de experimento de baixo nível
│   └── llm_factory.py       seleção entre LLM remoto, local e mock
│
├── evaluation/             métricas e benchmarks
│   ├── metrics.py           accuracy, acertos e comparação de assignments
│   └── greedy_mpe.py        benchmark MPE exato versus guloso
│
├── analysis/               código reutilizável de análise
│   ├── logs.py              leitura e normalização dos logs
│   ├── metrics.py           agregações por execução e variável
│   ├── structure.py         características estruturais das redes
│   ├── structural_metrics.py
│   ├── performance_plots.py
│   ├── structure_plots.py
│   └── style.py             identidade visual compartilhada
│
├── logging/                persistência dos resultados
├── io/                     carregamento de BIF, XDSL, NET, DSC e DNE
├── paths.py                caminhos canônicos dos artefatos
└── scripts/
    ├── run/                entry points de execução
    ├── analysis/           entry points de análise
    └── data_prep/          preparação pontual de datasets
```

Os módulos em `scripts/` devem apenas configurar e iniciar fluxos. Regras de
negócio, cálculos e componentes reutilizáveis ficam nos demais pacotes.

## Instalação

O projeto requer Python 3.12 ou superior e usa `uv` para gerenciamento do
ambiente.

```bash
uv sync --extra dev
```

Crie a configuração local:

```bash
cp .env.example .env
```

No Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

As configurações são lidas por `InferenceConfig` usando o prefixo `PGM_`.
As opções mais relevantes são:

```dotenv
PGM_LOG_FILE_NAME=logs/cbeb_evidence_position.jsonl
PGM_DEFAULT_ORDERING_HEURISTIC=min_degree
PGM_LLM_TIMEOUT=2400
PGM_LLM_MAX_RETRIES=3

PGM_OPENAI_API_KEY=...
PGM_OPENAI_BASE_URL=https://openrouter.ai/api/v1
PGM_OPENAI_MODEL=deepseek/deepseek-v4-flash

PGM_LOCAL_URL=http://localhost:11434/v1/chat/completions
PGM_LOCAL_MODEL=qwen3-coder:30b
```

Não versione o arquivo `.env` nem credenciais de API.

## Dados e artefatos

Os caminhos são definidos centralmente em `pgm_llm_inference.paths`:

| Conteúdo | Diretório |
|---|---|
| Redes Bayesianas | `src/datasets/` |
| Metadados de variáveis | `src/metadata/` |
| Notas de relacionamentos | `src/relationships/` |
| Compilações semânticas | `src/tables/` |
| Resultados dos experimentos | `logs/` |

O loader numérico aceita `.bif`, `.xdsl`, `.net`, `.dsc` e `.dne`, com ou
sem compactação `.gz`. O pipeline de compilação semântica trabalha com a
representação BIF utilizada pelos experimentos principais.

## Cache das compilações

Por padrão, cada dataset usa:

```text
src/tables/<dataset>.compiled.pkl
```

O parâmetro `model_name` já percorre o fluxo de cache. Em `mpe/cache.py` há
um switch comentado para usar nomes de arquivo separados por modelo durante
os experimentos.

O formato atual possui `COMPILED_SCHEMA_VERSION = 2`. Caches de versões
anteriores são considerados incompatíveis e fazem `load_or_compile` iniciar
uma nova compilação. Como essa operação pode consumir créditos do provedor,
preserve uma cópia dos caches antigos caso ainda precise executá-los com uma
versão anterior do código.

## Executando experimentos

Antes de executar, ajuste a lista de datasets e o `ExperimentConfig` no entry
point correspondente.

```bash
# Batch principal: múltiplos datasets e proporções de evidência
uv run python -m pgm_llm_inference.scripts.run.main

# Uma execução pequena com evidência fixa
uv run python -m pgm_llm_inference.scripts.run.main_single_run

# Experimento exaustivo de posição da evidência
uv run python -m pgm_llm_inference.scripts.run.run_evidence_pos

# Avaliação com rede sintética
uv run python -m pgm_llm_inference.scripts.run.synthetic_eval

# Benchmark MPE exato versus decodificação gulosa
uv run python -m pgm_llm_inference.scripts.run.get_mpe
```

## Demonstração sem CPTs

O exemplo mais direto de aplicação não precisa de arquivo BIF nem de tabelas
de probabilidade. Ele declara em Python os estados de cada variável, as arestas
no formato `pai -> filhos` e as evidências observadas:

```bash
uv run python -m pgm_llm_inference.scripts.run.cptless_demo
```

Para criar outra aplicação, edite apenas `VARIABLE_STATES`, `CHILDREN` e
`EVIDENCE` em `scripts/run/cptless_demo.py`. O modelo gera o contexto semântico,
compila as decisões e imprime uma tabela com o assignment completo. Como não
existem CPTs, o resultado é um MPE semântico aproximado e não pode ser tratado
como um MPE numérico exato.

### Modos de evidência

`ExperimentConfig.evidence_sampling` aceita:

- `mpe_consistent`: estados retirados do MPE incondicional;
- `mpe_inconsistent`: estados deliberadamente diferentes do MPE;
- `random`: estados amostrados aleatoriamente, sujeitos ao filtro de
  probabilidade configurado no batch.

### Seleção do LLM

Em `ExperimentConfig`:

| `use_real_llm` | `use_local_llm` | Cliente selecionado |
|---:|---:|---|
| `True` | qualquer valor | API configurada em `PGM_OPENAI_*` |
| `False` | `True` | servidor configurado em `PGM_LOCAL_*` |
| `False` | `False` | mock local |

## Analisando resultados

As análises consomem os logs JSONL produzidos pelos experimentos:

```bash
# Métricas agregadas, tabelas, correlações e gráficos
uv run python -m pgm_llm_inference.scripts.analysis.metrics

# Influência da profundidade do nó usado como evidência
uv run python -m pgm_llm_inference.scripts.analysis.evidence_position

# Segunda avaliação semântica por Markov Blanket
uv run python -m pgm_llm_inference.scripts.analysis.reinference

# Estatísticas estruturais e tabela LaTeX dos datasets
uv run python -m pgm_llm_inference.scripts.analysis.get_table
```

Os switches de gráficos ficam no `main()` de
`scripts/analysis/metrics.py`. É possível comentar ou descomentar as chamadas
sem alterar os módulos responsáveis pelos cálculos.

## Uso da inferência numérica

```python
import numpy as np

from pgm_llm_inference import (
    BayesianNetwork,
    Factor,
    InferenceEngine,
    SumProductStrategy,
    Variable,
)

rain = Variable(name="Rain", states=("yes", "no"))
network = BayesianNetwork(
    variables={"Rain": rain},
    factors=[Factor(scope=[rain], values=np.array([0.2, 0.8]))],
)

engine = InferenceEngine(network=network, strategy=SumProductStrategy())
result = engine.query(query_vars=["Rain"], evidence={})

print(result["result_factor"].values)
```

`Variable.states` é a representação canônica do domínio de uma variável. Não
há um segundo alias para os estados.

## Desenvolvimento e validação

```bash
uv run pytest -q
uv run ruff check src tests
uv run python -m compileall -q src tests
```

A suíte cobre carregamento de caminhos, parsing de respostas LLM, cache,
inferência compilada, execução dos experimentos e agregações de análise.

Após a refatoração arquitetural, a validação de referência é:

```text
20 testes aprovados
Ruff aprovado
compileall aprovado
smoke test numérico com asia.bif aprovado
```
