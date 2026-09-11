# Contexto do projeto

Este documento reúne o contexto científico e a terminologia necessários para
trabalhar no projeto sem reconstruir o domínio a cada tarefa. Para detalhes de
software, consulte [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. Problema investigado

Redes Bayesianas representam variáveis aleatórias discretas em um grafo
direcionado acíclico. Cada variável possui estados e, em uma rede tradicional,
uma tabela de probabilidade condicional (CPT) quantifica sua distribuição dado
o estado de seus pais.

O projeto investiga se um Large Language Model consegue substituir parte dessa
informação numérica por conhecimento semântico. A pergunta central é:

> Dada a estrutura de uma Rede Bayesiana, os estados possíveis e descrições
> qualitativas do domínio, um LLM consegue produzir configurações plausíveis
> que se aproximem do MPE obtido pelas CPTs reais?

O objetivo científico não é provar que texto equivale a probabilidade. O
objetivo é medir empiricamente a qualidade, estabilidade e limitações de uma
aproximação semântica quando comparada a uma referência numérica.

## 2. Dois regimes de uso

### Avaliação com CPTs

Nos experimentos principais, as CPTs existem, mas têm papéis separados:

- o pipeline numérico pode usá-las para calcular o Max-Product exato;
- o pipeline semântico não deve consultar seus valores;
- a saída exata serve como gabarito para avaliar a saída semântica.

Essa separação evita vazamento da resposta numérica para o método estudado.

### Aplicação sem CPTs

`scripts/run/cptless_demo.py` demonstra o caso em que existem apenas:

- nomes das variáveis;
- estados discretos;
- arestas do DAG;
- evidência opcional.

O resultado nesse regime é uma configuração semântica aproximada. Como não há
CPT, não existe um MPE numérico interno contra o qual provar optimalidade.

## 3. Terminologia

### Rede Bayesiana (BN)

Um DAG em que cada nó é uma variável aleatória e cada aresta representa uma
dependência direta. No código, é `BayesianNetwork`.

### Estado

Um valor discreto permitido para uma variável, por exemplo `low`, `medium` e
`high`. No código, a fonte canônica é `Variable.states`.

### CPT ou CPD

Tabela numérica de `P(X | Parents(X))`. Nos fatores internos, a variável filha
é o primeiro elemento do escopo e os pais vêm depois.

### Evidência

Assignment observado e fixo, como `{"Smoker": "yes"}`. Evidência não deve ser
alterada durante reconstrução.

### Variável oculta

Qualquer variável que não aparece na evidência.

### MAP

Maximum a Posteriori: assignment mais provável para um subconjunto de
variáveis de consulta, condicionado à evidência.

### MPE

Most Probable Explanation: assignment conjunto mais provável para todas as
variáveis não observadas, condicionado à evidência.

### Variable Elimination (VE)

Algoritmo que combina fatores e elimina variáveis. Sum-Product soma a variável
eliminada; Max-Product mantém o máximo e um backpointer para reconstrução.

### Compilação semântica

Fase offline em que o LLM recebe estrutura, metadados e contextos e escolhe um
estado para cada linha das tabelas de decisão.

### Inferência semântica

Fase online que fixa a evidência e consulta as tabelas já compiladas. Não chama
o LLM.

### `SemanticMessage`

Tabela produzida para uma variável. Cada `MessageRow` contém contexto,
`selected_value`, confiança e rationale. O nome histórico `llm_cpt` aparece em
logs, mas essas linhas não são probabilidades numéricas.

### Backpointer

Escolha registrada durante maximização. Na abordagem semântica, é o estado que
o LLM selecionou para um contexto específico.

### Markov Blanket

Pais, filhos e co-pais de uma variável. Condicionado ao seu Markov Blanket, o
nó é independente do restante da rede. O projeto o usa em análises e na etapa
de reinferência, não como parte da inferência compilada principal.

## 4. Método implementado

### 4.1 Conhecimento fornecido ao LLM

O LLM recebe uma combinação de:

- estrutura do DAG;
- nomes e estados das variáveis;
- papel topológico de cada nó;
- descrições, aliases e significados dos estados;
- notas qualitativas sobre relações entre pais e filhos;
- briefing global da rede;
- uma lista exata de contextos para os quais deve produzir decisões.

Metadados e notas podem existir em disco ou ser gerados pelo próprio LLM antes
da compilação das mensagens.

### 4.2 Compilação offline

A compilação ocorre com evidência vazia para tornar as tabelas reutilizáveis.
Para cada variável, o sistema enumera as configurações do separador e exige uma
resposta para cada uma. Respostas incompletas, estados ilegais e nomes
desconhecidos são rejeitados.

No algoritmo atual:

- a ordem de compilação é o reverso da ordem topológica;
- `active_messages` permanece vazio;
- o contexto de uma variável é essencialmente formado por seus pais;
- a escolha do LLM é registrada para cada configuração desses pais.

Portanto, a estrutura compilada atual se comporta como um conjunto de tabelas
de decisão semânticas condicionais.

### 4.3 Inferência online

A reconstrução percorre as variáveis em ordem causal:

1. se a variável é evidência, mantém o valor observado;
2. caso contrário, reúne os valores já atribuídos ao seu contexto;
3. procura a linha correspondente na mensagem;
4. atribui o `selected_value` registrado.

A mesma compilação responde a muitas evidências sem custo adicional de LLM.

### 4.4 Limite conceitual importante

Apesar da terminologia LLM-MPE e da inspiração em circuitos de maximização, o
código atual não implementa toda a propagação de fatores derivados de uma
bucket elimination numérica. Em particular, evidência em descendentes não
recalcula retroativamente as escolhas semânticas dos ancestrais.

Isso deve ser tratado como propriedade do método avaliado, não escondido pela
documentação. Implementar propagação bidirecional, reauditoria online ou novos
prompts seria uma nova versão do algoritmo e exigiria uma nova avaliação.

## 5. Referência numérica

O baseline exato usa Max-Product sobre as CPTs reais:

1. reduz fatores pela evidência;
2. multiplica fatores relevantes;
3. maximiza variáveis segundo uma ordem de eliminação;
4. registra argmax;
5. reconstrói o assignment conjunto.

Sum-Product também existe no motor, mas responde a consultas posteriores em
vez de produzir o MPE completo.

Há ainda um baseline guloso em `evaluation/greedy_mpe.py`: ele percorre a
ordem topológica e escolhe localmente `argmax P(X | parents(X))`. Esse baseline
ajuda a separar o ganho do método semântico do comportamento de uma simples
decodificação causal local.

## 6. Desenho dos experimentos

### Evidência MPE-consistente

Seleciona variáveis e usa os valores que elas possuem no MPE incondicional.
Esse cenário mede o comportamento quando a observação concorda com a
configuração global de referência.

### Evidência MPE-inconsistente

Seleciona valores diferentes daqueles do MPE incondicional. Esse cenário mede
robustez quando a observação força a solução para fora da configuração padrão.

### Evidência aleatória

Seleciona estados aleatórios, com possibilidade de rejeitar configurações cuja
probabilidade parcial seja muito baixa. É um cenário menos controlado.

### Proporção de evidência

Os batches variam o número de variáveis observadas. Para redes pequenas podem
ser usados todos os tamanhos; para redes maiores, `make_evidence_sizes` escolhe
pontos esparsos.

### Posição da evidência

O experimento `run_evidence_pos` usa cada variável, individualmente, como
evidência MPE-consistente e registra sua profundidade normalizada:

```text
0.0 = raiz
1.0 = nível mais profundo da rede
```

Isso permite estudar se evidências próximas às raízes ou às folhas afetam o
desempenho de forma diferente.

## 7. Métricas

### Accuracy por variável

Fração das variáveis avaliadas cujo estado semântico coincide com a referência
Max-Product.

### Exact match / joint match

Vale 1 somente quando todo o assignment avaliado coincide. É uma métrica mais
estrita que accuracy por variável.

### F1 macro e weighted

Tratam os estados previstos como classes. Macro dá o mesmo peso às classes;
weighted pondera pela frequência observada.

### Cohen's kappa

Mede concordância descontando a concordância esperada ao acaso. Pode ser
indefinido quando não há variação suficiente.

### Consenso

Para trials repetidos da mesma configuração, mede a proporção de votos da
predição modal por variável. Consenso alto mede estabilidade, não correção.

### Correlações estruturais

O projeto usa Spearman para relacionar desempenho com:

- número de nós e arestas;
- profundidade do DAG;
- cardinalidade média e máxima;
- número de pais por variável;
- cardinalidade por variável;
- posição normalizada da evidência.

No código atual, a coluna histórica `max_degree` calculada em `analysis.structure`
representa o maior número de pais de um nó, isto é, o grau de entrada máximo,
e não o grau total do grafo.

## 8. Datasets

Os datasets principais associados às análises CBEB são:

| Arquivo | Rótulo de domínio |
|---|---|
| `adhd_cbeb.bif` | TDAH |
| `alarm_cbeb.bif` | Monitoramento de UTI |
| `child_cbeb.bif` | Doenças pediátricas |
| `covid1_cbeb.bif` | Sintomas de Covid — 1 |
| `covid3_cbeb.bif` | Sintomas de Covid — 2 |
| `diabets_cbeb.bif` | Diabetes |
| `foodallergy1_cbeb.bif` | Alergia — 1 |
| `foodallergy3_cbeb.bif` | Alergia — 2 |
| `gonorrhoeae_cbeb.bif` | Gonorreia |
| `hepar2_cbeb.bif` | Hepatite |

O diretório `src/datasets/` também contém redes auxiliares e benchmarks. A
presença de um arquivo no diretório não significa que ele pertença ao protocolo
experimental atual; as listas efetivamente executadas ficam nos entry points.

Nomes de arquivos, variáveis e estados fazem parte da identidade experimental.
Renomeá-los exige atualizar metadados, relacionamentos, caches e logs.

## 9. Artefatos produzidos

### Metadados

Descrições por variável, incluindo display name, descrição, expert note,
aliases e significado dos estados.

### Notas de relacionamento

Orientações qualitativas sobre mecanismos entre uma variável e seus pais,
incluindo direção, modulação e armadilhas de raciocínio.

### Compilação

Um pickle `CompiledSemanticMessages` com tudo o que a fase online precisa. O
pickle inclui traces das chamadas, úteis para auditoria.

### Logs

JSONL com evidência, predições, referência, confiança, métricas, dataset,
modelo e campos específicos do experimento.

## 10. Reprodutibilidade

Para comparar execuções, registre ou preserve:

- commit do código;
- dataset exato;
- metadados e notas de relacionamento;
- arquivo compilado e sua versão de schema;
- provedor e nome do modelo;
- temperatura e parâmetros do cliente;
- prompt vigente;
- seed e estratégia de amostragem;
- configuração do experimento;
- log bruto antes das agregações.

Reutilizar uma compilação elimina novas chamadas durante a inferência, mas não
remove a dependência das escolhas feitas pelo modelo durante a compilação.

## 11. Hipóteses e limitações

- Nomes e descrições carregam conhecimento suficiente para escolhas úteis.
- Um LLM pode produzir decisões semanticamente coerentes sem ler CPTs.
- A qualidade depende do modelo, prompt, metadados e domínio.
- Confiança textual do LLM não é probabilidade calibrada.
- Accuracy alta não implica recuperação do assignment conjunto.
- Consenso alto pode refletir erro sistemático.
- Contextos crescem com o produto das cardinalidades do separador.
- Datasets e número de redes limitam o poder das correlações em nível de rede.
- O regime sem CPT não permite medir optimalidade exata internamente.
- O algoritmo online atual favorece fluxo causal de pais para filhos e não
  realiza atualização retroativa completa a partir de descendentes.

Essas limitações orientam a interpretação dos resultados e não devem ser
“corrigidas” sem redefinir a hipótese científica e repetir os experimentos.

## 12. Documentos conceituais históricos

Os arquivos `semantic-maximizing-circuits.md` e `online-phase-mpe.md` descrevem
a motivação baseada em circuitos de maximização e uma formulação conceitual de
propagação online. Eles são úteis como histórico teórico, mas este documento e
o código são a referência para o comportamento atualmente implementado.
