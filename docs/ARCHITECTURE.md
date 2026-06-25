# Arquitetura do Sistema

Documenta o schema do grafo heterogêneo, o feature engineering de cada nó, a arquitetura interna dos três modelos GNN, e as decisões de design que motivaram cada escolha.

---

## Índice

1. [Schema do Grafo](#schema-do-grafo)
2. [Feature Engineering](#feature-engineering)
3. [Normalização e Target](#normalização-e-target)
4. [Modelos GNN](#modelos-gnn)
5. [Decisões de Design](#decisões-de-design)
6. [Arquivos Críticos](#arquivos-críticos)

---

## Schema do Grafo

O grafo é **heterogêneo** — tem múltiplos tipos de nós e arestas com semânticas diferentes. Isso é representado como `HeteroData` no PyTorch Geometric.

### Tipos de Nós

| Nó | Representa | Quantidade (PoC) | Quantidade (major) |
|----|-----------|-----------------|-------------------|
| `Flight` | Um voo individual (rota + data + horário) | 2.360 | ~357K |
| `Airport` | Aeroporto (identificado pelo código ICAO) | 1.252 | ~170 |
| `Aircraft` | Tipo de aeronave × companhia (ex: GOL-B738) | 58 | ~200 |
| `Clima` | Condição meteorológica horária em um aeroporto | 2.294 | ~180K |

> **Importante:** `Aircraft` representa o **tipo de equipamento por companhia**, não a aeronave individual. `GOL-B738` e `LATAM-B738` são nós diferentes, pois têm frotas e históricos distintos.

### Tipos de Arestas

| Aresta | Direção | Significado |
|--------|---------|------------|
| `ORIGIN` | Flight → Airport | Aeroporto de origem do voo |
| `DESTINATION` | Flight → Airport | Aeroporto de destino do voo |
| `NEXT_LEG` | Flight → Flight | Mesmo número de voo em dias consecutivos (mesma rota recorrente) |
| `NEXT_ROTATION` | Flight → Flight | Próximo voo do mesmo tipo de aeronave no mesmo dia |
| `ASSIGNED_TO` | Flight → Aircraft | Tipo de aeronave designado para o voo |
| `HAS_ORIGIN_WEATHER` | Flight → Clima | Clima no aeroporto de origem no horário do voo |
| `OBSERVED_AT` | Clima → Airport | Observação meteorológica pertence a qual aeroporto |

Após aplicar `ToUndirected()`, cada aresta ganha uma versão reversa com prefixo `rev_`. Assim, **7 tipos originais → 12 tipos de aresta** no total (algumas já são bidirecionais por natureza e não geram reversa duplicada).

### Por que esse schema captura o atraso?

O grafo modela explicitamente os **três principais vetores de propagação de atraso**:

1. **Propagação temporal (NEXT_LEG / NEXT_ROTATION):** um avião que chegou atrasado provavelmente sairá atrasado no próximo voo. O `NEXT_LEG` captura padrões de rota (a LA3001 de segunda frequentemente atrasa porque a de domingo também atrasa). O `NEXT_ROTATION` captura a propagação direta dentro do dia.

2. **Condições no aeroporto de origem (HAS_ORIGIN_WEATHER → OBSERVED_AT):** vento forte, chuva e nebulosidade aumentam o risco de atraso. O clima é conectado ao voo *no horário* do voo, não genericamente.

3. **Características estruturais (ORIGIN / DESTINATION → Airport):** aeroportos em regiões de altitude elevada ou latitude extrema têm mais variabilidade climática. Aeroportos congestionados (SBGR, SBSP) têm padrões de atraso distintos.

---

## Feature Engineering

### Flight (3 features)

| Feature | Como é calculada | Por que importa |
|---------|-----------------|----------------|
| `hour_dep` | Hora UTC da partida programada (0–23) | Voos de madrugada e cedo tendem a ter menos atraso que os da tarde (efeito cascata ao longo do dia) |
| `duration_min` | `(sched_arr - sched_dep).total_seconds() / 60` | Voos mais longos têm mais oportunidade de recuperar atraso em rota |
| `delay_dep_min_clipped` | `delay_departure_min` clipado em `[-30, 300]` | Atraso na partida é o preditor mais forte do atraso na chegada |

**Tratamento de NaN:** preenchido com a mediana de cada coluna antes de normalizar.

---

### Airport (3 features)

| Feature | Fonte | Por que importa |
|---------|-------|----------------|
| `latitude` | `airports.csv` (OurAirports) | Aeroportos em latitudes extremas têm mais variabilidade climática |
| `longitude` | `airports.csv` | Identifica região geográfica (Norte, Sul, litoral, interior) |
| `elevation` | `airports.csv` (pés) | Altitude alta aumenta risco de neblina/gelo. SBCF (Confins) fica a 800m, impacta operações |

**Tratamento:** convertidos para numérico, NaN preenchido com 0.

---

### Aircraft (2 features)

| Feature | Como é calculada | Por que importa |
|---------|-----------------|----------------|
| `generation_age_years` | Idade média do tipo ICAO na frota brasileira (ANAC RAB). Fallback: `CURRENT_YEAR - ano_de_introducao_do_tipo` | Aeronaves mais antigas têm mais manutenções imprevistas |
| `is_low_cost` | 1.0 se a airline está na lista `LOW_COST_AIRLINES`, 0.0 caso contrário | Companhias low-cost operam com menos margem de tempo entre voos (mais suscetíveis a efeito cascata) |

**Baixa cardinalidade:** o nó Aircraft representa o tipo × companhia, então há poucos nós (~50-200). Suas features chegam aos voos via message passing pelas arestas `ASSIGNED_TO`.

---

### Clima (4 features)

| Feature | Fonte | Por que importa |
|---------|-------|----------------|
| `temp` | Open-Meteo | Temperatura extrema afeta performance da aeronave |
| `windspeed` | Open-Meteo | Vento forte aumenta tempo de taxiamento e pode causar desvios |
| `rain` | Open-Meteo | Chuva reduz visibilidade, aumenta separação entre pousos |
| `clouds` | Open-Meteo | Nebulosidade (0-100%) indica visibilidade geral |

A granularidade é **horária por aeroporto** — cada voo é vinculado ao clima do aeroporto de origem *na hora* do voo programado. Isso é feito pelo script `link_clima.py` que cria as arestas `HAS_ORIGIN_WEATHER`.

---

## Normalização e Target

### Features

Todas as features numéricas passam por **z-score normalização** (`StandardScaler` do scikit-learn):

```
X_normalizado = (X - média) / desvio_padrão
```

O `StandardScaler` é fitado no conjunto completo de dados (não só no treino), pois o objetivo é representação consistente dos nós, não prevenir data leakage no target.

### Target (`delay_arrival_min`)

O target passa por duas etapas antes da normalização:

**1. Clip:** `delay_arrival_min` é clipado em `[-30, 360]` minutos.
- `-30 min` como limite inferior: atrasos "negativos" (chegadas adiantadas) raramente passam de 30 min e têm semântica diferente
- `360 min` (6 horas) como limite superior: atrasos extremos (cancelamentos que viraram voos, dados errados) são outliers que prejudicam o treino sem representar o padrão geral

**2. Z-score:** depois do clip, aplica z-score com a média e desvio padrão dos dados clipados.

Os valores `y_mean` e `y_std` são salvos como atributos do `HeteroData` para que `predict.py` e `train_minibatch.py` possam denormalizar as previsões em minutos reais.

**Loss:** MSE sobre o target normalizado. Isso equivale a minimizar o erro relativo ao desvio padrão dos atrasos — mais interpretável que MSE em minutos brutos.

**Métricas reportadas:** MAE e RMSE são sempre **denormalizados** (em minutos reais) para facilitar interpretação.

---

## Modelos GNN

Todos os modelos usam o mesmo framework: message passing heterogêneo com 2 camadas convolucionais + head de regressão linear. A diferença está em **como** cada camada agrega mensagens dos vizinhos.

### Modelo 1 — GAT (Graph Attention Network)

**Classe:** `FlightDelayGAT` em `train.py`

```
Entrada:  x_dict (features por tipo de nó)
          edge_index_dict (arestas por tipo)

Camada 1: HeteroConv({ tipo_aresta: GATConv(in=-1, out=64, heads=4, concat=True) })
          → saída: 64 × 4 = 256 dims por nó (concatena as 4 cabeças)
          → ELU → Dropout(0.3)

Camada 2: HeteroConv({ tipo_aresta: GATConv(in=-1, out=64, heads=4, concat=False) })
          → saída: 64 dims por nó (média das 4 cabeças)
          → ELU

Head:     Linear(64 → 32) → ReLU → Dropout(0.3) → Linear(32 → 1)
Saída:    predição escalar por nó Flight
```

**`HeteroConv`**: wrapper do PyG que aplica uma convolução independente para cada tipo de aresta e soma os resultados. Cada tipo de aresta tem seu próprio conjunto de pesos de atenção.

**`GATConv`**: para cada nó destino, calcula um peso de atenção para cada vizinho e faz a agregação como média ponderada. Vizinhos mais relevantes recebem peso maior. Com `in=-1`, as dimensões de entrada são inferidas automaticamente no primeiro forward pass (inicialização lazy).

**Por que `concat=True` na camada 1 e `concat=False` na 2?**
- Camada 1: concatenar as cabeças aumenta a capacidade de representação (256 dims vs 64). Útil nas camadas iniciais para capturar features diversas.
- Camada 2: fazer média reduz a dimensão de volta para 64, preparando para o head de regressão. Evita explodir parâmetros.

**Resultado:** MAE 25,92 min (melhor dos 3 modelos, converge em ~36 épocas).

---

### Modelo 2 — HGT (Heterogeneous Graph Transformer)

**Classe:** `FlightDelayHGT` em `train.py`

```
Entrada:  x_dict, edge_index_dict, metadata (lista de tipos de nó e aresta)

Projeção: Linear(in_feat → 64) por tipo de nó → ReLU
          (necessário: HGTConv exige dimensão comum entre tipos de nó)

Camada 1: HGTConv(64, 64, metadata, heads=4) → ELU → Dropout(0.3)
Camada 2: HGTConv(64, 64, metadata, heads=4) → ELU

Head:     Linear(64 → 32) → ReLU → Dropout(0.3) → Linear(32 → 1)
```

**`HGTConv`**: baseado no paper "Heterogeneous Graph Transformer" (Hu et al. 2020). Diferentemente do GAT, mantém **matrizes de transformação separadas para cada combinação de tipo de nó e tipo de aresta**. Isso permite que o modelo aprenda semânticas específicas: "como um Airport influencia um Flight via ORIGIN" é diferente de "como um Airport influencia um Flight via DESTINATION".

**Por que a projeção inicial?** O HGTConv requer que todos os tipos de nó tenham a mesma dimensão de embedding. Como cada tipo tem um número diferente de features (Flight=3, Airport=3, Aircraft=2, Clima=4), a projeção linear unifica tudo em 64 dims.

**Resultado:** MAE 28,42 min, apenas 181K parâmetros (8× menos que o GAT). A convergência é mais lenta (ainda melhorando na época 80), o que sugere que com mais épocas poderia igualar o GAT.

---

### Modelo 3 — TGN (Temporal Graph Network — adaptado)

**Classe:** `FlightDelayTGN` em `train.py`

```
Entrada:  x_dict, edge_index_dict

Projeção: Linear(in_feat → 64) por tipo de nó → ReLU

Memória:  GRUCell(64, 64) aplicado na ordem topológica das cadeias NEXT_LEG
          Para cada voo na cadeia: h_t = GRU(h_{t-1}, x_projetado)
          Início de cadeia (sem predecessor): h_0 = zeros

Residual: flight_h = flight_h_projetado + memory (soma com output da GRU)

Conv:     HeteroConv({ tipo_aresta: GATConv(in=-1, out=64, heads=4, concat=False) })
          → ELU

Head:     Linear(64 → 32) → ReLU → Dropout(0.3) → Linear(32 → 1)
```

**Motivação:** o TGN original (Rossi et al. 2020) mantém uma memória por nó que é atualizada à medida que eventos ocorrem. Aqui, adaptamos a ideia para o contexto heterogêneo: a cadeia `NEXT_LEG` é essencialmente uma sequência temporal de voos da mesma rota. A GRU processa essa sequência, acumulando o "estado de saúde" da rota ao longo do tempo.

**Por que teve resultado ruim (MAE 77,55 min)?**
- **Vanishing gradient**: cadeias de 400+ voos (uma aeronave operando por 15 meses) fazem os gradientes desaparecerem nas primeiras posições da sequência. A GRU aprende bem os voos recentes, mas "esquece" os antigos.
- **Processamento sequencial**: a GRU não é paralelizável ao longo da cadeia, tornando o treino muito mais lento.
- **Adaptação parcial**: o TGN original usa um módulo de embedding, função de mensagem e módulo de memória separados. Nossa adaptação simplificada perde parte dessa expressividade.

**Possíveis melhorias:** truncar as cadeias (ex: máximo 30 voos), usar learning rate menor, ou substituir GRU por Transformer (que paraleliza).

---

## Decisões de Design

### Por que `ToUndirected()`?

O grafo original tem arestas direcionadas (ex: `Flight → Airport`). Isso significa que durante o message passing, um `Airport` pode influenciar um `Flight` só via `rev_ORIGIN`, mas o `Flight` não enviaria mensagem de volta para o `Airport` via `ORIGIN`.

`ToUndirected()` dobra todas as arestas adicionando versões reversas. Isso permite **fluxo bidirecional de informação**: um `Airport` congestionado "avisa" os voos que partem dele, e esses voos "atualizam" o estado do aeroporto com informações sobre atrasos observados.

A decisão foi movida do `train.py` para o `build_graph.py` para que o grafo salvo em disco já contenha as arestas reversas — evita reprocessamento a cada run de treino.

---

### Por que z-score e não min-max?

Min-max normalização é sensível a outliers: um atraso de 1000 minutos comprimiria todos os outros valores próximos de 0. O dataset tem distribuição assimétrica (muitos voos pontuais, poucos com atraso extremo). Z-score é mais robusto a essa assimetria.

Além disso, o clip em `[-30, 360]` antes da normalização remove os outliers mais extremos antes do z-score, resultando em uma distribuição bem comportada para o treino.

---

### Por que split aleatório e não temporal?

Para o TCC, a escolha foi pragmática: com dados de Jan/2025 a Mar/2026, um split temporal (ex: treinar até Dez/2025, testar em 2026) seria mais realista para avaliação de produção, mas torna difícil comparar arquiteturas de forma controlada.

O split aleatório (70/15/15) garante que todas as épocas e rotas estejam representadas em treino, validação e teste — mais adequado para comparação de arquiteturas. Para um sistema em produção, o split temporal seria o correto.

---

### Por que `equipment_icao` como tipo e não matrícula individual?

A API VRA/ANAC não fornece matrícula individual de forma consistente em todos os registros. O `equipment_icao` (tipo do avião: B738, A320, E195...) é o campo mais completo e confiável disponível. A consequência é que "Aircraft" no grafo representa o perfil operacional de um tipo de aeronave por companhia, não uma aeronave específica.

Isso é academicamente defensável: aeronaves do mesmo tipo têm comportamento similar, e a granularidade por (airline × tipo) ainda captura diferenças operacionais importantes.

---

### Full-batch vs Mini-batch

| Critério | Full-batch (`train.py`) | Mini-batch (`train_minibatch.py`) |
|----------|------------------------|----------------------------------|
| RAM durante treino | Alta (grafo inteiro em RAM) | Menor (só o batch atual no forward pass) |
| Convergência | Mais rápida e estável | Mais lenta (ruído do sampling) |
| Qualidade (PoC) | MAE 25,92 min | MAE 26,43 min |
| Escalabilidade | Limitado pelo tamanho do grafo | Escala para grafos maiores |

Para o dataset completo (814K voos, `graph.pt` ~218 MB), o mini-batch é necessário se a RAM disponível for insuficiente para o full-batch. Para datasets menores, o full-batch é preferível.

---

## Arquivos Críticos

| Arquivo | Responsabilidade |
|---------|-----------------|
| `build_graph.py` | Neo4j → HeteroData PyG. Feature engineering, normalização, `ToUndirected`. Classe `GraphExtractor`. |
| `train.py` | Definição dos 3 modelos (GAT, HGT, TGN) + loop de treino full-batch. Único ponto de verdade para arquitetura dos modelos. |
| `train_minibatch.py` | Loop de treino mini-batch com `NeighborLoader`. Mesmos modelos do `train.py`. |
| `predict.py` | Inferência. Importa `build_model` de `train.py` e funções de feature engineering de `build_graph.py`. |
| `update_graph.py` | Atualização incremental do grafo. Mantém consistência de normalização com o grafo original. |
| `load_data.py` | Ingestão inicial. Classe `Neo4jDB` com todos os métodos de escrita no banco. |
| `enrich_aircraft_age.py` | Migração + enriquecimento dos nós Aircraft com dados do ANAC RAB. |
| `create_next_rotation.py` | Criação das arestas NEXT_ROTATION por tipo de aeronave × dia. |
| `api_calls.py` | Wrappers para APIs externas (VRA/ANAC, OpenSky, Open-Meteo) com retry e backoff. |
