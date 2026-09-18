# Instruções para agentes

Este arquivo define como agentes de código devem trabalhar neste repositório.
Leia também [ARCHITECTURE.md](docs/ARCHITECTURE.md) e
[PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) antes de modificar o pipeline.

## 1. Prioridades

Ao tomar decisões, use esta ordem:

1. preservar a validade dos resultados científicos;
2. não consumir créditos nem sobrescrever artefatos sem autorização;
3. preservar o comportamento solicitado;
4. respeitar as fronteiras arquiteturais;
5. manter o código simples, explícito e testável.

Não trate uma alteração de algoritmo, prompt, amostragem ou métrica como mera
refatoração. Essas mudanças podem invalidar comparações experimentais.

## 2. Antes de começar

1. Leia o `README.md` e os documentos desta pasta.
2. Execute `git status --short` e considere toda alteração existente como
   pertencente ao usuário.
3. Localize chamadores com `rg` antes de mover, renomear ou apagar símbolos.
4. Para mudanças no fluxo principal, comece em `scripts/run/main.py` e siga as
   chamadas até o logger.
5. Identifique se a tarefa é numérica, semântica, experimental ou analítica.
6. Declare qualquer suposição que possa afetar resultados ou artefatos.

Não reverta nem formate indiscriminadamente mudanças que não fazem parte da
tarefa atual.

## 3. Proteção de custos e dados

As seguintes ações podem chamar um LLM real e consumir créditos:

- executar `scripts.run.main` sem um cache válido;
- executar `scripts.run.run_evidence_pos` sem as compilações necessárias;
- executar `scripts.run.cptless_demo`;
- chamar `compile_semantic_messages` ou `load_or_compile` em condições de cache
  miss;
- gerar novamente metadados ou notas de relacionamentos.

Portanto:

- nunca use esses fluxos como smoke test sem autorização explícita;
- prefira testes com mocks para clientes LLM;
- não apague nem mova `src/tables/`, `src/metadata/`, `src/relationships/`,
  `logs/` ou datasets sem autorização;
- não abra, imprima nem versione `.env` ou chaves de API;
- faça backup antes de migrar pickles ou logs científicos;
- lembre que cache incompatível inicia recompilação automaticamente.

O comentário de switch em `mpe/cache.py`, que alterna cache por dataset ou por
dataset e modelo, é intencional e não deve ser removido.

## 4. Fronteiras de módulos

### `models/`

Contém apenas os modelos canônicos:

- `Variable.states` é a única representação do domínio;
- `Factor.scope[0]` é a variável filha e os demais elementos são pais;
- `BayesianNetwork` serve tanto ao pipeline numérico quanto ao semântico;
- `BayesianNetwork.from_structure` cria redes sem CPTs e valida o DAG.

Não crie versões duplicadas desses modelos em outros pacotes.

### `core/`

Contém operações matemáticas, ordenação e configuração de runtime.
Não deve conhecer scripts, relatórios ou desenho de experimentos.

### `inference/`

`engine.py` reúne Variable Elimination, reconstrução numérica e helpers de
inferência exata. `strategies.py` define as operações de eliminação. Evite
branches de estratégia dentro do motor quando o comportamento puder permanecer
polimórfico.

### `io/` e `llm/`

`io/loaders.py` reúne carregamento e conversão para os modelos canônicos.
`llm/providers.py` seleciona os clientes de metadados e relacionamentos;
`llm/parsing.py` trata suas respostas. O cliente especializado da compilação
fica em `mpe/compile_phase/client.py`, com seu próprio contrato de parsing e
retentativas. `io.loaders.parse_bif` lê apenas a estrutura para o pipeline
semântico, mantendo as CPTs fora desse caminho.

### `mpe/`

Contém compilação e inferência semânticas. Preserve a separação:

- `pre_compile_phase/` prepara metadados, notas e briefing;
- `compile_phase/` constrói, compila, valida e propaga mensagens dos buckets;
- `compile.py` coordena as fases e mantém `CompiledSemanticMessages` no caminho
  usado pelos pickles; os schemas serializados permanecem em `types.py`;
- `infer.py` faz lookup e reconstrução; normalização e chaves de contexto ficam
  em `normalization.py`, sem depender das fases de preparação e compilação;
- `state_semantics.py` é compartilhado pelo briefing e pelos prompts dos buckets;

- compilação pode chamar LLM;
- inferência compilada não chama LLM;
- reconstrução é determinística;
- compilação ocorre com evidência vazia e propaga mensagens entre buckets;
- alterações em prompts ou schemas são mudanças de comportamento;
- alterações serializadas exigem avaliar `COMPILED_SCHEMA_VERSION`.

### `experiment/`

Contém configuração, amostragem, batches, comparação entre métodos e logging
JSONL. Não mova cálculos puros de fatores ou plotting para essa camada.

### `evaluation/` e `analysis/`

`evaluation/` mede resultados durante ou próximo da execução. `analysis/`
processa logs, estrutura e visualizações pós-hoc. Funções estatísticas devem ser
separadas de impressão e plotting sempre que possível.

### `scripts/`

Scripts são entry points. Eles podem conter listas de datasets, switches
experimentais e um `main()`, mas não lógica reutilizável. É proibido criar
dependências script → script; extraia a função compartilhada para a camada
adequada.

## 5. Convenções de código

- Python 3.12 ou superior.
- Use type hints modernos: `list[str]`, `dict[str, str]`, `X | None`.
- Use `pathlib.Path` para caminhos.
- Importe caminhos canônicos de `pgm_llm_inference.paths`.
- Prefira funções pequenas e dados explícitos a wrappers e adapters.
- Não adicione `sys.path` hacks.
- Não duplique configurações existentes.
- Não use argumentos mantidos apenas por compatibilidade sem um caso atual.
- Não capture `Exception` em código de domínio sem acrescentar contexto ou uma
  política clara; tolerância ampla é aceitável em loops de CLI que precisam
  continuar para o próximo dataset.
- Preserve nomes de variáveis e estados exatamente como definidos no dataset.
- Mantenha cálculos puros independentes de `print`, arquivos e gráficos.
- Use inglês para identificadores e APIs; mensagens e documentação podem ser
  em português.
- Comentários devem explicar decisões e invariantes, não narrar cada linha.
- Evite marcadores históricos como `[NOVO]`, `[MODIFICADO]` e código comentado
  sem uma finalidade experimental explícita.

## 6. Mudanças que exigem cuidado científico

Considere as seguintes alterações como potencialmente incompatíveis:

- prompt, temperatura, modelo ou parser de resposta;
- ordem de eliminação;
- composição do contexto de um bucket;
- propagação de mensagens entre buckets;
- tratamento de evidência;
- critério de voto, agrupamento ou denominador de métricas;
- geração de evidência e seeds;
- significado dos campos do log;
- schema de `CompiledSemanticMessages`;
- mapeamento de nomes ou estados dos datasets.

Nesses casos, documente antes/depois, acrescente testes e informe que resultados
antigos podem deixar de ser diretamente comparáveis.

## 7. Processo recomendado para mudanças

1. Descreva o problema concreto e confirme seus chamadores.
2. Escolha a camada responsável antes de editar.
3. Faça uma mudança coesa por vez.
4. Acrescente ou adapte testes na mesma etapa.
5. Rode a validação completa.
6. Revise `git diff --check` e `git status --short`.
7. Atualize documentação quando caminhos, APIs, formatos ou comportamento
   mudarem.

Para remover código morto, uma ausência de import no `__init__.py` não basta.
Procure referências em `src`, `tests`, scripts, README e documentos. Considere
também entry points executados diretamente.

## 8. Comandos de validação

Com `uv`:

```bash
uv run pytest -q -p no:cacheprovider
uv run ruff check src tests
uv run python -m compileall -q src tests
git diff --check
```

No ambiente virtual do Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
```

Use `-p no:cacheprovider` quando o diretório `.pytest_cache` estiver com ACL
restritiva. Isso não altera o resultado dos testes.

Não rode automaticamente `ruff format` sobre o repositório inteiro: ele pode
misturar uma mudança funcional pequena com dezenas de alterações mecânicas.

## 9. Estratégia de testes

- Modelos: validação de estados, fatores, arestas e ciclos.
- Core numérico: arrays pequenos com resultado conhecido.
- Pipeline semântico: clientes mockados e respostas Pydantic controladas.
- Cache: diretório temporário e mocks; nunca use as tabelas reais nos testes.
- Experimentos: mock de Max-Product e inferência compilada.
- Análise: DataFrames pequenos e determinísticos.
- Caminhos: compare objetos `Path`, sem depender do diretório corrente.

Um smoke test numérico com um dataset pequeno é aceitável porque não chama LLM.
Confirme que o caminho exercitado não passa por compilação semântica.

## 10. Checklist de entrega

- [ ] O comportamento solicitado está implementado, não apenas descrito.
- [ ] Não há dependência de uma camada inferior para `scripts/`.
- [ ] Não há import entre scripts.
- [ ] Nenhum segredo ou artefato científico foi incluído no diff.
- [ ] Nenhum fluxo real de LLM foi executado sem autorização.
- [ ] Mudanças de schema de cache foram tratadas explicitamente.
- [ ] Testes novos cobrem o risco principal da mudança.
- [ ] Ruff, pytest, compileall e `git diff --check` passam.
- [ ] README e documentos continuam coerentes com o código.

## 11. Estado de referência

No encerramento da refatoração arquitetural:

- o fluxo principal começa em `scripts/run/main.py`;
- cálculos reutilizáveis não vivem em scripts;
- caminhos estão centralizados;
- compilação e inferência semântica estão separadas;
- o cache está versionado;
- a API pública de `mpe` expõe compilação, tipo compilado e inferência;
- redes sem CPT podem ser construídas com `BayesianNetwork.from_structure`.

Se o código divergir deste estado, atualize este documento deliberadamente.
