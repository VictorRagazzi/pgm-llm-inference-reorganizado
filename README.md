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

Durante a compilação com evidência vazia, mensagens são propagadas entre
buckets e podem ampliar seus separadores além dos pais. A inferência posterior
consulta essas decisões sem recalcular mensagens em função da evidência.

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
│   ├── factor_ops.py       produto, redução, soma e maximização de fatores
│   └── ordering.py         heurísticas de ordem de eliminação
│
├── inference/              motor numérico de Variable Elimination
│   ├── engine.py           VE, reconstrução numérica e referência exata
│   └── strategies.py       estratégias Sum-Product e Max-Product
│
├── llm/                    clientes de metadados, notas e análises auxiliares
│   ├── providers.py         seleção e chamadas aos modelos remoto/local
│   └── parsing.py           parsing das respostas desses clientes
│
├── mpe/                    pipeline semântico LLM-MPE
│   ├── compile.py           coordenação das fases e tipo compilado
│   ├── cache.py             persistência e versionamento da compilação
│   ├── infer.py             lookup e reconstrução determinística
│   ├── graph.py             operações de grafo
│   ├── normalization.py     nomes, estados e chaves de contexto
│   ├── state_semantics.py   interpretação compartilhada dos estados
│   ├── types.py             schemas do pipeline
│   ├── pre_compile_phase/
│   │   ├── metadata.py      carregamento e geração de metadados
│   │   ├── relationships.py carregamento e geração de notas
│   │   └── briefing.py      contexto semântico e briefing da rede
│   └── compile_phase/
│       ├── buckets.py       construção, compilação, validação e propagação
│       ├── prompts.py       prompts dos buckets
│       ├── client.py        cliente com validação, retentativas e log-probs
│       └── domain_scoring.py códigos categóricos dos estados
│
├── experiment/             execução dos experimentos
│   ├── config.py            ExperimentConfig e modos MAP/MPE
│   ├── batch.py             geração do batch
│   ├── sampling.py          estratégias de amostragem de evidência
│   ├── runner.py            comparação exata versus semântica
│   └── logging.py           persistência JSONL
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
│   ├── domain_entropy.py
│   ├── token_vectors.py
│   ├── entropy_plots.py
│   └── style.py             identidade visual compartilhada
│
├── io/                     carregamento e conversão de BIF, XDSL, NET, DSC e DNE
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
sem compactação `.gz`. O pipeline de compilação semântica usa
`io.loaders.parse_bif`, que lê somente nomes, estados e topologia do BIF,
sem carregar as probabilidades numéricas.

## Cache das compilações

Na configuração ativa, cada par de dataset e modelo usa:

```text
src/tables/<dataset>.<modelo>_VE.compiled.pkl
```

O parâmetro `model_name` percorre o fluxo de cache. Em `mpe/cache.py` há um
switch comentado para voltar temporariamente a um arquivo único por dataset;
esse comentário é intencional.

O formato atual possui `COMPILED_SCHEMA_VERSION = 5`. Caches de versões
anteriores são considerados incompatíveis e fazem `load_or_compile` iniciar
uma nova compilação. Como essa operação pode consumir créditos do provedor,
preserve uma cópia dos caches antigos caso ainda precise executá-los com uma
versão anterior do código.

Durante a compilação de variáveis ocultas, cada estado recebe um código curto
(`A`, `B`, `C`...) usado apenas na resposta da LLM. Os log-probs desses códigos
são convertidos novamente para os nomes canônicos de `Variable.states` e
armazenados por context row em `MessageRow.domain_scores`. As alternativas
brutas do token de decisão (até 20 pela API compatível) ficam separadamente em
`MessageRow.token_scores`. Quando o modelo ou
provedor não fornece log-probs, o campo permanece opcional e o restante do
fluxo não muda. Resultados produzidos antes da versão 5 não são diretamente
comparáveis a esses scores categóricos.

Para inspecionar todas as context rows de um pickle novo, sem chamar a LLM:

```bash
uv run python -m pgm_llm_inference.scripts.analysis.decision_entropy \
  --compiled src/tables/<dataset>.<modelo>_VE.compiled.pkl \
  --output-dir plots/decision_entropy
```

O CSV e o gráfico distinguem entropia do domínio, entropia condicional das
alternativas top-k e massa de probabilidade coberta pelo top-k. A segunda é
truncada e não representa a entropia do vocabulário completo. Mesmo para uma
variável binária, a entropia do domínio já descreve toda a incerteza entre os
dois estados; os tokens extras servem para diagnosticar incerteza de geração
e formatação, não para ampliar o domínio da variável.

Para gerar somente os vetores brutos, um PNG por variável e um CSV com cada
token/context row:

```bash
uv run python -m pgm_llm_inference.scripts.analysis.decision_entropy \
  --compiled src/tables/<dataset>.<modelo>_VE.compiled.pkl \
  --output-dir figs --token-vectors-only
```

Os PNGs ficam em `figs/token_vectors/`. Para variáveis com várias context rows,
cada linha representa uma delas e cada coluna uma alternativa ordenada pelo
log-prob retornado. Células vazias
indicam tokens não retornados, não probabilidade zero. Os vetores incluem o
top-k disponível (até 20 pela API atual) e o token gerado se ele ficou fora
do top-k; não são o vocabulário completo do modelo. O script recusa sobrescrever
arquivos existentes e avisa quando o pickle não contém `token_scores`.

## Executando experimentos

Antes de executar, ajuste a lista de datasets e o `ExperimentConfig` no entry
point correspondente.

```bash
# Batch principal: múltiplos datasets e proporções de evidência
uv run python -m pgm_llm_inference.scripts.run.main

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

Para geração de metadados e notas, `ExperimentConfig` seleciona:

| `use_real_llm` | `use_local_llm` | Cliente selecionado |
|---:|---:|---|
| `True` | qualquer valor | API configurada em `PGM_OPENAI_*` |
| `False` | `True` | servidor configurado em `PGM_LOCAL_*` |
| `False` | `False` | mock local |

Briefing e decisões dos buckets usam `mpe.compile_phase.client.LLMJsonClient`: remoto quando
`use_real_llm=True`, local caso contrário. O mock acima não substitui esse
cliente; os testes simulam os dois caminhos separadamente.

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
uv run pytest -q -p no:cacheprovider
uv run ruff check src tests
uv run python -m compileall -q src tests
git diff --check
```

A suíte usa redes sintéticas, arquivos temporários e clientes LLM simulados.
Ela não lê `.env` e bloqueia requisições HTTP. Cobre a referência numérica por
enumeração, carregamento de formatos, contratos dos clientes, cache,
compilação, reconstrução, batches e logs JSONL.
Há também um teste de carregamento de pickle e inferência em um processo novo,
que verifica que as fases de preparação e compilação não são importadas.

`tests/fixtures/pipeline.json` contém a referência capturada antes da
simplificação: hashes de 8 prompts, mensagens compiladas, resultados de
inferência, 63 configurações de batch e registros JSONL. Não regenere esse
arquivo para acomodar diferenças sem verificar sua causa.

A simplificação preserva o comando principal, as repetições, seeds,
retentativas e métricas. Também preserva os caminhos das classes serializadas
e o schema de cache 5; não exige migração nem recompilação. Os imports internos
consolidados estão documentados em [ARCHITECTURE.md](docs/ARCHITECTURE.md).
