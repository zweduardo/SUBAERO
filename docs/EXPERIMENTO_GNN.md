# Experimento: Comparacao de Arquiteturas GNN para Predicao de Atraso de Voos

**Data:** 2026-03-19
**Projeto:** TCC MBA IA & Big Data
**Objetivo:** Comparar 3 arquiteturas de Graph Neural Networks na predicao de atraso de chegada de voos brasileiros

---

## 1. Contexto

### Problema
Prever o atraso de chegada (em minutos) de voos comerciais brasileiros utilizando um grafo heterogeneo que modela as relacoes entre voos, aeroportos, aeronaves e condicoes climaticas.

### Dataset
- **Fonte:** Neo4j (dados da API VRA/ANAC + Open-Meteo)
- **Periodo:** Janeiro/2025 a Marco/2026
- **Total no banco:** ~814.000 voos
- **Amostra PoC:** 2.360 voos (6 aeronaves selecionadas)
- **Amostragem:** Por cadeia de aeronave (preserva arestas NEXT_LEG)

### Grafo Heterogeneo

**4 tipos de no:**

| No | Qtd (PoC) | Features | Descricao |
|---|---|---|---|
| Flight | 2.360 | 3 (hour_dep, duration_min, delay_dep_min_clipped) | Voo individual |
| Airport | 1.252 | 3 (latitude, longitude, elevation) | Aeroporto |
| Aircraft | 58 | 1 (dummy — placeholder) | Tipo de aeronave (ICAO) |
| Clima | 2.294 | 4 (temp, windspeed, rain, clouds) | Condicao meteorologica horaria |

**6 tipos de aresta (11 com reversas via ToUndirected):**

| Aresta | Direcao | Qtd (PoC) | Significado |
|---|---|---|---|
| ORIGIN | Flight -> Airport | 2.360 | Aeroporto de origem |
| DESTINATION | Flight -> Airport | 2.360 | Aeroporto de destino |
| NEXT_LEG | Flight -> Flight | 1.984 | Proximo voo da mesma aeronave (propagacao de atraso) |
| ASSIGNED_TO | Flight -> Aircraft | 2.360 | Aeronave designada |
| HAS_ORIGIN_WEATHER | Flight -> Clima | 2.360 | Clima na origem no horario do voo |
| OBSERVED_AT | Clima -> Airport | 2.294 | Observacao climatica no aeroporto |

### Target
- **Variavel:** `delay_arrival_min` (atraso de chegada em minutos)
- **Normalizacao:** z-score (clip em [-30, 360] min antes de normalizar)
- **Metricas:** MAE e RMSE em minutos reais (denormalizados)

### Split dos Dados
- **Treino:** 70% (1.652 voos)
- **Validacao:** 15% (354 voos)
- **Teste:** 15% (354 voos)
- **Metodo:** Split aleatorio por indice (transductivo — todos os nos presentes no grafo durante treino)

---

## 2. Arquiteturas Comparadas

### 2.1 GAT (Graph Attention Network)

Rede de atencao sobre grafos heterogeneos usando `HeteroConv` com `GATConv` por tipo de aresta.

**Arquitetura:**
```
Camada 1: HeteroConv(GATConv(in=-1, out=64, heads=4, concat=True))  -> ELU -> Dropout(0.3)
Camada 2: HeteroConv(GATConv(in=-1, out=64, heads=4, concat=False)) -> ELU
Head:     Linear(64 -> 32) -> ReLU -> Dropout(0.3) -> Linear(32 -> 1)
```

**Caracteristicas:**
- Cada tipo de aresta tem pesos de atencao independentes
- `concat=True` na camada 1 (saida = 64 x 4 = 256 dims), `concat=False` na camada 2 (media das cabecas = 64 dims)
- Lazy initialization (`in_channels=(-1,-1)`) — infere dimensoes automaticamente
- **1.475.585 parametros treinaveis**

### 2.2 HGT (Heterogeneous Graph Transformer)

Transformer projetado especificamente para grafos heterogeneos (`HGTConv`), com transformacoes e atencao especificas por tipo de no e aresta.

**Arquitetura:**
```
Projecao:  Linear(in_feat -> 64) por tipo de no -> ReLU
Camada 1:  HGTConv(64, 64, metadata, heads=4) -> ELU -> Dropout(0.3)
Camada 2:  HGTConv(64, 64, metadata, heads=4) -> ELU
Head:      Linear(64 -> 32) -> ReLU -> Dropout(0.3) -> Linear(32 -> 1)
```

**Caracteristicas:**
- Projecao linear por tipo de no para dimensao comum (necessario pelo HGTConv)
- Atencao multi-head com matrizes de peso especificas por tipo de no E tipo de aresta
- Baseado no paper "Heterogeneous Graph Transformer" (HGT, Hu et al. 2020)
- **181.345 parametros treinaveis** (8x menos que GAT)

### 2.3 TGN (Temporal Graph Network — adaptado)

Modelo inspirado no TGN (Rossi et al. 2020), adaptado para o contexto heterogeneo. O TGN original opera sobre `TemporalData` (grafos de interacao temporal). Aqui adaptamos as ideias centrais para `HeteroData`:

**Arquitetura:**
```
Projecao:     Linear(in_feat -> 64) por tipo de no -> ReLU
Memoria GRU:  GRUCell(64, 64) propagada na ordem topologica das cadeias NEXT_LEG
Residual:     flight_h = flight_h + memory (soma features projetadas com memoria)
Convolucao:   HeteroConv(GATConv(in=-1, out=64, heads=4, concat=False)) -> ELU
Head:         Linear(64 -> 32) -> ReLU -> Dropout(0.3) -> Linear(32 -> 1)
```

**Caracteristicas:**
- **Memoria temporal:** GRU processa voos em ordem topologica ao longo das cadeias NEXT_LEG, simulando o modulo de memoria do TGN
- No inicio de cada cadeia (sem predecessor), memoria inicializada com zeros
- Conexao residual: embedding final = features projetadas + estado da memoria
- Convolucao espacial agrega contexto de aeroporto, clima e aeronave
- **394.817 parametros treinaveis**
- Processamento sequencial da GRU (nao paralelizavel) torna o treino mais lento

---

## 3. Configuracao do Experimento

| Parametro | Valor |
|---|---|
| Epocas | 80 |
| Learning rate | 1e-3 |
| Otimizador | Adam (weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Hidden dim | 64 |
| Attention heads | 4 |
| Dropout | 0.3 |
| Gradient clipping | max_norm=1.0 |
| Loss | MSE (sobre target normalizado em z-score) |
| Hardware | CPU (sem GPU) |
| ToUndirected | Sim (arestas reversas adicionadas para fluxo bidirecional) |

---

## 4. Resultados

### Metricas Finais (conjunto de teste)

| Modelo | Parametros | Teste MAE (min) | Teste RMSE (min) | Melhor Epoca |
|---|---|---|---|---|
| **GAT** | 1.475.585 | **25,92** | **34,50** | 36 |
| **HGT** | 181.345 | 28,42 | 37,46 | 78 |
| **TGN** | 394.817 | 77,55 | 101,81 | 80 |

### Curva de Convergencia (Val MAE por epoca)

**GAT:**
```
Epoca  1: 115.12    Epoca 10:  46.84    Epoca 20:  28.56
Epoca 30:  27.82    Epoca 40:  26.31    Epoca 50:  26.29
Epoca 60:  25.89    Epoca 70:  25.74    Epoca 80:  25.80
-> Convergiu na epoca 36, estabilizou depois
```

**HGT:**
```
Epoca  1: 114.84    Epoca 10: 115.47    Epoca 20: 113.89
Epoca 30: 109.68    Epoca 40: 100.74    Epoca 50:  82.05
Epoca 60:  57.51    Epoca 70:  35.06    Epoca 80:  28.92
-> Convergencia lenta, ainda melhorando na epoca 80
```

**TGN:**
```
Epoca  1: 113.93    Epoca 10: 100.45    Epoca 20:  76.93
Epoca 30:  75.90    Epoca 40:  71.35    Epoca 50:  71.71
Epoca 60:  70.66    Epoca 70:  70.43    Epoca 80:  70.02
-> Convergencia rapida ate epoca 30, depois estagnacao
```

---

## 5. Analise

### GAT — Melhor Desempenho
- Convergencia rapida (melhor resultado na epoca 36)
- Maior numero de parametros devido ao `concat=True` na primeira camada (multiplica dims por numero de cabecas)
- O mecanismo de atencao por aresta e eficiente para capturar a importancia relativa de cada relacao
- O scheduler reduziu o LR apos a epoca 40, ajudando a estabilizar

### HGT — Eficiente e Promissor
- 8x menos parametros que o GAT, resultado apenas 10% pior
- Convergencia lenta nas primeiras 40 epocas — tipico do HGT que precisa aprender transformacoes especificas por tipo
- Ainda estava melhorando na epoca 80, sugerindo que com mais epocas pode igualar ou superar o GAT
- Projetado especificamente para grafos heterogeneos — aproveita a semantica dos tipos

### TGN — Limitacoes na Adaptacao
- O loop sequencial da GRU sobre cadeias de ~400 voos por aeronave causa:
  - **Vanishing gradients** em cadeias longas
  - **Treino lento** (nao paralelizavel)
- Estagnacao apos epoca 30 (MAE ~70 min) indica dificuldade de otimizacao
- A adaptacao do TGN para HeteroData perde parte da arquitetura original (message function, embedding module, interaction graphs)
- Com truncamento de cadeias, learning rate menor, ou mais dados, poderia melhorar

### Observacoes Gerais
- A amostra de 2.360 voos (de 6 aeronaves) tem distribuicao enviesada: 68% dos voos com atraso >15min, media de atraso de 269.9 min
- Isso difere do dataset completo (22% atrasados, media 13.5 min), impactando a generalizacao
- O NEXT_LEG e a aresta mais importante — sem ela, o MAE piora significativamente (demonstrado em teste anterior)

---

## 6. Funcionalidades Implementadas

### Treino Incremental (Partial Fit)
O pipeline suporta adicao de novos dados sem retreinar do zero:

```bash
# 1. Adiciona voos novos ao grafo (marca com new_mask)
python update_graph.py --since 2026-03-01 --graph graph.pt --output graph.pt

# 2. Fine-tune: carrega pesos existentes, loss so nos voos novos
python train.py --model gat --graph graph.pt --finetune model.pt --output model.pt --epochs 20
```

O grafo completo (antigo + novo) e usado para message passing, mas o loss e calculado somente nos voos novos. Assim o modelo se adapta sem retreinar nos 814k+ voos anteriores.

### Predicao de Voos Futuros
```bash
# Busca voos futuros no Neo4j (sem delay_arrival_min), conecta ao grafo, preve
python predict.py --graph graph.pt --model model.pt --model-type gat --future-since 2026-03-20

# Voos especificos
python predict.py --graph graph.pt --model model.pt --model-type gat --flights "FLT001,FLT002"
```

Para voos futuros, `delay_departure_min` e desconhecido (usa 0). A GNN infere o atraso provavel via:
- **NEXT_LEG:** propagacao do atraso da aeronave anterior
- **Clima:** condicoes meteorologicas no aeroporto de origem
- **Aeroporto:** caracteristicas estruturais (latitude, altitude)

---

## 7. Reproducao

### Ambiente
- Python 3.11 (Anaconda)
- PyTorch + PyTorch Geometric
- Neo4j 5.x (bolt://192.168.15.118:7687)
- Hardware: CPU (sem GPU)

### Comandos
```bash
# Construir grafo PoC (amostragem por aeronave)
python build_graph.py --sample 2000 --output graph_poc.pt --stats

# Treinar os 3 modelos
python train.py --model gat --graph graph_poc.pt --output model_gat.pt --epochs 80
python train.py --model hgt --graph graph_poc.pt --output model_hgt.pt --epochs 80
python train.py --model tgn --graph graph_poc.pt --output model_tgn.pt --epochs 80

# Previsoes
python predict.py --graph graph_poc.pt --model model_gat.pt --model-type gat --output predictions.csv
```

### Seed
- Amostragem de aeronaves: `random_state=42`
- Split treino/val/teste: `random_state=42`
