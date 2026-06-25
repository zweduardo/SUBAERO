# Guia de Execução do Pipeline

Este documento descreve **cada fase do pipeline**, o que cada script faz internamente, todos os parâmetros disponíveis e o que esperar de cada execução.

---

## Índice

1. [Fase 0 — Setup](#fase-0--setup)
2. [Fase 1 — Ingestão de Dados no Neo4j](#fase-1--ingestão-de-dados-no-neo4j)
3. [Fase 2 — Build do Grafo PyG](#fase-2--build-do-grafo-pyg)
4. [Fase 3 — Treino do Modelo GNN](#fase-3--treino-do-modelo-gnn)
5. [Fase 4 — Inferência](#fase-4--inferência)
6. [Fase 5 — Update Incremental](#fase-5--update-incremental)
7. [Referência Rápida de Comandos](#referência-rápida-de-comandos)

---

## Fase 0 — Setup

### Dependências Python

```bash
pip install torch torch-geometric neo4j pandas numpy scikit-learn requests
```

> Não existe `requirements.txt` no projeto. Instalar manualmente no ambiente Anaconda.

### Neo4j

O Neo4j precisa estar rodando localmente antes de qualquer script do pipeline. A conexão está hardcoded em todos os scripts:

```python
NEO4J_URI      = "bolt://192.168.15.118:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "tcc12345"
```

Se o Neo4j estiver em outro endereço, editar diretamente em cada arquivo.

### Credenciais OpenSky

A API OpenSky usa OAuth2. As credenciais ficam em `credentials.json` na raiz do projeto:

```json
{
  "client_id": "seu_client_id",
  "client_secret": "seu_client_secret"
}
```

> **Nunca commitar esse arquivo.** Ele está no `.gitignore`.

---

## Fase 1 — Ingestão de Dados no Neo4j

Esta fase popula o banco Neo4j com todos os dados brutos. Deve ser executada **uma única vez** na carga inicial. Depois, use o `update_graph.py` (Fase 5) para dados incrementais.

### 1.1 — Carga inicial completa

```bash
python load_data.py
```

**O que faz internamente:**
- Baixa dados de voos da **API VRA/ANAC** (todos os voos comerciais com atraso registrado)
- Cria nós `Flight` com: `flight_id`, `scheduled_departure`, `scheduled_arrival`, `delay_departure_min`, `delay_arrival_min`, `airline_code`, `flight_number`, `equipment_icao`
- Cria nós `Airport` a partir do `airports.csv` (fonte: OurAirports), filtrados para `large_airport` no Brasil
- Cria nós `Aircraft` por tipo de equipamento ICAO (ex: B738, A320) — representa o **tipo**, não matrícula individual
- Cria arestas `ORIGIN`, `DESTINATION`, `ASSIGNED_TO`
- Cria arestas `NEXT_LEG` conectando voos consecutivos da mesma rota (mesmo número de voo em dias diferentes)
- Baixa dados meteorológicos do **Open-Meteo** para cada aeroporto e cria nós `Clima` + arestas `HAS_ORIGIN_WEATHER` e `OBSERVED_AT`

**Tempo estimado:** horas (depende do volume de dados e limites de API).

**Constraints criadas no Neo4j:**
```cypher
UNIQUE: Airport.icao_code
UNIQUE: Aircraft.node_key
INDEX:  Airport.iata_code
INDEX:  Flight.flight_id
INDEX:  Flight.scheduled_departure
INDEX:  Clima.clima_id
INDEX:  Flight.(airline_code, flight_number)
```

---

### 1.2 — Enriquecer nós Aircraft com idade da frota

```bash
python enrich_aircraft_age.py
python enrich_aircraft_age.py --dry-run   # só mostra o que faria, sem gravar
```

**O que faz internamente:**

1. **Migração dos nós Aircraft**: reestrutura de "por tipo ICAO" para "por (airline_code × equipment_icao)". Isso porque uma mesma aeronave B738 operada pela GOL e pela LATAM tem contextos diferentes (frota, manutenção, modelo de negócio).

2. **Consulta ao ANAC RAB** (Registro Aeronáutico Brasileiro): baixa o CSV público com a frota ativa brasileira (`dados_aeronaves.csv`) e calcula a **idade média real** de cada tipo de aeronave.

3. **Fallback estático**: se o tipo não estiver no RAB, usa o dicionário `TYPE_INTRO_YEAR` com o ano de introdução de cada modelo (B738 → 1998, A320 → 1988, etc.) e calcula `CURRENT_YEAR - intro_year`.

4. **Grava no Neo4j** as propriedades:
   - `generation_age_years`: idade média do tipo em anos
   - `is_low_cost`: 1.0 se a airline é low-cost (GOL, Azul, etc.), 0.0 caso contrário
   - `node_key`: chave única `airline_code_equipment_icao`

**Por que isso importa:** aeronaves mais antigas tendem a ter mais problemas mecânicos, manutenções imprevistas e consequentemente mais atrasos.

---

### 1.3 — Criar arestas NEXT_ROTATION

```bash
python create_next_rotation.py
python create_next_rotation.py --dry-run              # conta sem criar
python create_next_rotation.py --since 2026-01-01    # só voos após a data
```

**O que faz internamente:**

Para cada tipo de equipamento (ex: todos os voos com B738), ordena os voos do dia por horário de partida programada e conecta cada voo ao próximo do mesmo tipo de aeronave no mesmo dia:

```
Voo A (B738, 06:00) -[NEXT_ROTATION]-> Voo B (B738, 09:00) -[NEXT_ROTATION]-> Voo C (B738, 14:00)
```

**Diferença crucial entre NEXT_LEG e NEXT_ROTATION:**

| Aresta | Conecta | Captura |
|--------|---------|---------|
| `NEXT_LEG` | Mesmo número de voo em dias diferentes (ex: LA3001 de segunda → LA3001 de terça) | Padrões sazonais de rota |
| `NEXT_ROTATION` | Voos consecutivos do mesmo tipo de aeronave no mesmo dia | Propagação de atraso (avião chegou tarde → próximo voo atrasa) |

**Por que usa tipo e não matrícula:** os dados VRA/ANAC não têm matrícula individual confiável em todos os registros. O `equipment_icao` (tipo) é o campo mais completo disponível.

O script processa em **batches de 500 equipamentos** por transação Neo4j para não sobrecarregar o banco.

---

### 1.4 — Utilitários de Clima (uso eventual)

```bash
# Recarregar dados meteorológicos horários do Open-Meteo
python reload_clima_hourly.py

# Recriar arestas HAS_ORIGIN_WEATHER (Flight -> Clima) após recarga
python link_clima.py
```

Usar quando os dados de clima ficarem desatualizados ou se a vinculação entre voos e climas precisar ser refeita.

---

## Fase 2 — Build do Grafo PyG

```bash
python build_graph.py [opções]
```

Este script conecta ao Neo4j, extrai todos os dados, faz feature engineering e serializa um objeto `HeteroData` do PyTorch Geometric em disco.

### Parâmetros disponíveis

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `--output` | string | `data/graph.pt` | Caminho do arquivo de saída |
| `--sample` | int | — | Modo PoC: amostra ~N voos por **cadeia de aeronave** (preserva NEXT_LEG) |
| `--filter-airports` | flag | desligado | Filtra voos onde origem E destino estão em `MAJOR_AIRPORTS` (~14 principais aeroportos BR) |
| `--since` | data ISO | — | Inclui apenas voos com `scheduled_departure >= data` |
| `--until` | data ISO | — | Inclui apenas voos com `scheduled_departure < data` |
| `--stats` | flag | desligado | Imprime estatísticas do grafo (contagem de nós, arestas, features) |

### Quando usar cada variante

**Teste rápido / PoC:**
```bash
python build_graph.py --sample 2000 --output data/graph_poc.pt --stats
```
Amostra ~2.000 voos selecionando aeronaves inteiras (preserva a cadeia NEXT_LEG). Roda em segundos. Ideal para testar código, debugar, experimentar hiperparâmetros.

**Dataset reduzido (economiza RAM, treino viável na maioria dos PCs):**
```bash
python build_graph.py --filter-airports --output data/graph_major.pt --stats
```
Filtra apenas voos entre os ~14 aeroportos de maior tráfego no Brasil (ex: SBGR, SBSP, SBGL, SBBR...). Resulta em ~357K voos vs ~814K do dataset completo. O arquivo `data/graph_major.pt` ocupa ~58 MB vs ~218 MB do completo.

**Janela temporal:**
```bash
python build_graph.py --since 2025-07-01 --until 2026-01-01 --output data/graph_2h2025.pt --stats
```
Útil para treinar em um período específico ou para análises temporais.

**Dataset completo:**
```bash
python build_graph.py --output data/graph.pt --stats
```
Todos os ~814K voos. Arquivo de ~218 MB. Requer bastante RAM para treinar (full-batch).

### O que acontece internamente

O script executa 3 etapas principais:

**[1/3] Extração dos nós:**
- Busca `Flight` com `delay_arrival_min IS NOT NULL` (voos com target conhecido)
- Se `--since` passado: também busca "âncoras NEXT_LEG" — voos anteriores ao período que servem como ponto de entrada para as cadeias de atraso
- Busca todos os `Airport`, `Aircraft` e `Clima` referenciados

**[2/3] Extração das arestas:**
- Para cada tipo de aresta, executa uma query Cypher e retorna DataFrame com `(src_id, dst_id)`
- A função `edge_tensor()` converte para tensor PyTorch usando lookup vetorizado com numpy (sem `iterrows`, ~10-20× mais rápido)
- Arestas cujo nó source ou destination não esteja no conjunto selecionado são ignoradas silenciosamente

**[3/3] Construção do HeteroData:**
- Features de cada nó são z-score normalizadas com `StandardScaler`
- Target (`delay_arrival_min`) é clipado em `[-30, 360]` minutos antes de normalizar — remove outliers extremos
- `ToUndirected()` é aplicado: dobra todas as arestas adicionando versões reversas (ex: `ORIGIN` → `rev_ORIGIN`). Isso permite que a informação flua nos dois sentidos durante o message passing
- `y_mean` e `y_std` são salvos como atributos do grafo para denormalizar as previsões depois

### Arquivo de saída (`data/graph.pt`)

O arquivo é um `HeteroData` serializado com `torch.save()`. Contém:

```
graph["flight"].x           # tensor [N_voos, 3] — features dos voos
graph["flight"].y           # tensor [N_voos]    — target normalizado
graph["flight"].y_raw       # tensor [N_voos]    — target original (minutos)
graph["flight"].y_mean      # float              — média usada na normalização
graph["flight"].y_std       # float              — desvio padrão usado
graph["flight"].flight_ids  # list[str]          — IDs dos voos (para debug)
graph["airport"].x          # tensor [N_airports, 3]
graph["airport"].icao       # list[str]          — códigos ICAO
graph["aircraft"].x         # tensor [N_aircraft, 2]
graph["clima"].x            # tensor [N_clima, 4]
graph[("flight","ORIGIN","airport")].edge_index         # tensor [2, E]
graph[("flight","DESTINATION","airport")].edge_index
graph[("flight","NEXT_LEG","flight")].edge_index
graph[("flight","NEXT_ROTATION","flight")].edge_index
graph[("flight","ASSIGNED_TO","aircraft")].edge_index
graph[("flight","HAS_ORIGIN_WEATHER","clima")].edge_index
graph[("clima","OBSERVED_AT","airport")].edge_index
# + versões rev_* de cada aresta (adicionadas pelo ToUndirected)
```

---

## Fase 3 — Treino do Modelo GNN

Existem **dois scripts de treino** com abordagens diferentes:

| Script | Abordagem | Quando usar |
|--------|-----------|-------------|
| `train.py` | Full-batch | Dataset pequeno/médio, melhor convergência |
| `train_minibatch.py` | Mini-batch (NeighborLoader) | Dataset grande, memória limitada |

### 3.1 — train.py (Full-batch)

```bash
python train.py --model gat --graph data/graph.pt --output models/model.pt --epochs 80
```

**Full-batch significa:** a cada época, o grafo **inteiro** passa pelo modelo de uma vez. Gradientes calculados sobre todos os voos. Mais estável e converge mais rápido, mas requer que o grafo inteiro caiba na RAM.

#### Parâmetros

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--model` | obrigatório | Arquitetura: `gat`, `hgt` ou `tgn` |
| `--graph` | `data/graph.pt` | Arquivo do grafo (gerado pelo build_graph.py) |
| `--output` | `models/model.pt` | Onde salvar os pesos do modelo treinado |
| `--epochs` | `80` | Número de épocas de treino |
| `--hidden` | `64` | Dimensão dos embeddings ocultos |
| `--heads` | `4` | Número de attention heads (GAT e HGT) |
| `--dropout` | `0.3` | Taxa de dropout durante treino |
| `--lr` | `1e-3` | Learning rate inicial |
| `--wd` | `1e-4` | Weight decay (regularização L2) |
| `--finetune` | — | Caminho para pesos existentes (modo fine-tune) |
| `--log-every` | `5` | Frequência de log no terminal (épocas) |

#### O que acontece internamente

1. Carrega `data/graph.pt` na RAM
2. Verifica se o grafo já tem arestas reversas (`rev_*`). Se não tiver (grafo antigo), aplica `ToUndirected()` automaticamente — compatibilidade com grafos gerados antes da v2 do `build_graph.py`
3. Cria masks de treino/validação/teste: **70% / 15% / 15%** (split aleatório, seed=42)
4. Instancia o modelo escolhido com inicialização lazy (dimensões inferidas no primeiro forward pass)
5. Loop de treino:
   - Forward: calcula embeddings para todos os nós via message passing
   - Loss: MSE sobre o target normalizado (só nos voos do `train_mask`)
   - Backward + Adam optimizer
   - Validation MAE (denormalizado em minutos reais)
   - `ReduceLROnPlateau`: reduz LR pela metade se Val MAE não melhorar em 10 épocas
   - Salva o modelo sempre que Val MAE melhora
6. Avaliação final no conjunto de teste com o melhor modelo
7. Salva métricas por época em `<output>.csv`

#### Exemplo de saída esperada

```
Dispositivo: cpu
Modelo: GAT

Carregando 'graph_poc.pt'...
  2360 voos  |  12 tipos de aresta
  Split - treino: 1652  val: 354  teste: 354
  Parametros treinaveis: 1,475,585

Treinando por 80 epocas...

 Epoca      Loss   Val MAE   Val RMSE         LR
-------------------------------------------------
     1    0.9231    115.12     135.40   1.00e-03
     5    0.4821     46.84      62.30   1.00e-03
    10    0.3102     28.56      40.11   1.00e-03
    ...
    36    0.1854     24.98      35.02   1.00e-03  ← melhor
    ...
    80    0.1823     25.80      35.33   3.13e-05

Melhor modelo: epoca 36  val MAE = 24.98 min

----------------------------------------
  [GAT] Teste - MAE  : 25.92 min
  [GAT] Teste - RMSE : 34.50 min
----------------------------------------
```

---

### 3.2 — train_minibatch.py (Mini-batch)

```bash
python train_minibatch.py --model gat --graph data/graph.pt --epochs 80 --output models/model.pt
```

**Mini-batch significa:** a cada época, o grafo é dividido em batches de voos. Para cada batch, o `NeighborLoader` amostra uma vizinhança local (K vizinhos por hop) e processa só esse subgrafo. Reduz o pico de RAM **durante o forward pass**, mas o grafo completo ainda precisa ser carregado em memória.

> **Quando usar:** dataset grande (`data/graph.pt` completo com 814K voos) onde o full-batch estouraria a RAM durante o treino. Para datasets menores (`data/graph_poc.pt`, `data/graph_major.pt`), o `train.py` é preferível — convergência mais rápida e MAE ligeiramente melhor.

#### Parâmetros adicionais (além dos do train.py)

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--batch-size` | `512` | Voos por batch |
| `--num-neighbors` | `10,5` | Vizinhos amostrados por hop (camada 1, camada 2) |
| `--device` | auto | Forçar backend: `cpu`, `cuda`, `directml` |

#### Diferença de qualidade

Nos testes com `data/graph_poc.pt` (2.360 voos, 80 épocas):

| | Full-batch | Mini-batch |
|--|-----------|-----------|
| Teste MAE | 25,92 min | 26,43 min |
| Teste RMSE | 34,50 min | 35,16 min |

Qualidade praticamente idêntica. A pequena diferença vem do fato que o NeighborLoader amostra apenas K vizinhos por hop — não vê o grafo completo em cada atualização.

---

### 3.3 — Modo Fine-tune

Para reaproveitamento de pesos treinados com novos dados (Fase 5):

```bash
python train.py --model gat --graph data/graph_updated.pt --finetune models/model.pt --output models/model.pt --epochs 20
```

**O que muda:** se o grafo tiver atributo `new_mask` (criado pelo `update_graph.py`), o loss é calculado **apenas nos voos novos** (marcados como `True` no mask). O message passing ainda usa o grafo completo (antigos + novos), então os embeddings dos novos voos se beneficiam do contexto dos voos já treinados. Só os gradientes que afetam voos novos são considerados no loss.

Isso evita retreinar nos ~800K voos anteriores quando chega uma nova batch de dados.

---

## Fase 4 — Inferência

```bash
python predict.py --graph data/graph.pt --model models/model.pt --model-type gat [opções]
```

### Modos de operação

**Modo 1 — Todos os voos do grafo:**
```bash
python predict.py --graph data/graph.pt --model models/model.pt --model-type gat
```
Prevê para todos os voos presentes em `data/graph.pt`. Útil para análise geral ou auditoria do modelo.

**Modo 2 — Voos específicos por ID:**
```bash
python predict.py --graph data/graph.pt --model models/model.pt --model-type gat --flights "FLT001,FLT002,FLT003"
```
Filtra a predição para uma lista de `flight_id` separados por vírgula.

**Modo 3 — Voos futuros (sem atraso registrado):**
```bash
python predict.py --graph data/graph.pt --model models/model.pt --model-type gat --future-since 2026-03-20
```
Busca no Neo4j voos com `delay_arrival_min IS NULL` (ainda não aconteceram ou não foram registrados) e `scheduled_departure >= data`. Adiciona esses voos temporariamente ao grafo existente e faz predição.

> Para voos futuros, `delay_departure_min` é desconhecido — o modelo usa 0. A GNN infere o atraso via `NEXT_LEG` (se a aeronave anterior já está em atraso no grafo) e via clima/aeroporto.

**Salvar resultado em CSV:**
```bash
python predict.py --graph data/graph.pt --model models/model.pt --model-type gat --output results/predictions.csv
```

### Parâmetros

| Parâmetro | Obrigatório | Descrição |
|-----------|------------|-----------|
| `--graph` | sim | Arquivo do grafo |
| `--model` | sim | Arquivo dos pesos treinados |
| `--model-type` | sim | Arquitetura: `gat`, `hgt` ou `tgn` |
| `--flights` | não | Lista de flight_ids separados por vírgula |
| `--future-since` | não | Busca voos futuros a partir dessa data (ISO) |
| `--output` | não | Salva CSV com colunas: `flight_id, predicted_delay_min` |

### O que acontece internamente

1. Carrega `data/graph.pt` e o modelo treinado
2. Se `--future-since`: conecta ao Neo4j, busca voos sem target, cria features (usando `delay_dep_min=0`), os anexa ao grafo como novos nós com arestas relevantes
3. Forward pass do modelo em modo `eval()` (sem dropout, sem gradientes)
4. Denormaliza as previsões: `pred_minutos = pred_normalizado × y_std + y_mean`
5. Exibe tabela ou salva CSV

---

## Fase 5 — Update Incremental

Quando chegam novos dados (ex: voos de um novo mês), não é necessário reconstruir o grafo do zero nem retreinar do zero.

### Passo 1 — Atualizar o grafo

```bash
python update_graph.py --since 2026-03-01 --graph data/graph.pt --output data/graph_updated.pt
```

**O que faz:**
1. Carrega o `data/graph.pt` existente
2. Busca no Neo4j voos com `scheduled_departure >= --since` que **não estão** no grafo atual
3. Calcula features dos novos voos usando a mesma normalização do grafo original (usa o `y_mean` e `y_std` já gravados — importante para consistência)
4. Anexa os novos nós `Flight` e `Clima` ao grafo
5. Cria todas as arestas relevantes dos novos voos (ORIGIN, DESTINATION, NEXT_LEG, ASSIGNED_TO, HAS_ORIGIN_WEATHER)
6. Marca os novos voos com `new_mask = True` — flag que o `train.py` usa no fine-tune

**Parâmetros:**

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--since` | obrigatório | Data mínima dos novos voos (ISO: `2026-03-01`) |
| `--graph` | `data/graph.pt` | Grafo existente a ser atualizado |
| `--output` | igual a `--graph` | Onde salvar o grafo atualizado (pode sobrescrever) |

### Passo 2 — Fine-tune do modelo

```bash
python train.py --model gat --graph data/graph_updated.pt --finetune models/model.pt --output models/model.pt --epochs 20
```

O modelo carrega os pesos existentes e treina apenas com loss nos voos novos (`new_mask = True`). Menos épocas são necessárias porque o modelo já tem bom conhecimento base — 20 épocas costuma ser suficiente.

### Por que isso funciona

Durante o fine-tune, o message passing percorre o **grafo completo** (antigos + novos), então os embeddings dos novos voos são enriquecidos pelo contexto dos voos antigos. Mas só os novos voos contribuem para o loss e para os gradientes. É como se o modelo "aprendesse" os padrões dos novos dados sem esquecer os antigos.

---

## Referência Rápida de Comandos

```bash
# ─── Ingestão (uma vez) ────────────────────────────────────────────────────
python load_data.py
python enrich_aircraft_age.py
python create_next_rotation.py

# ─── PoC (rápido, para testes) ────────────────────────────────────────────
python build_graph.py --sample 2000 --output data/graph_poc.pt --stats
python train.py --model gat --graph data/graph_poc.pt --output models/model_poc.pt --epochs 80
python predict.py --graph data/graph_poc.pt --model models/model_poc.pt --model-type gat

# ─── Dataset major airports (~357K voos) ──────────────────────────────────
python build_graph.py --filter-airports --output data/graph_major.pt --stats
python train.py --model gat --graph data/graph_major.pt --output models/model_major.pt --epochs 80

# ─── Dataset completo (~814K voos, exige RAM) ─────────────────────────────
python build_graph.py --output data/graph.pt --stats
python train_minibatch.py --model gat --graph data/graph.pt --output models/model.pt --epochs 80

# ─── Comparar os 3 modelos ────────────────────────────────────────────────
python train.py --model gat --graph data/graph_poc.pt --output models/model_gat.pt --epochs 80
python train.py --model hgt --graph data/graph_poc.pt --output models/model_hgt.pt --epochs 80
python train.py --model tgn --graph data/graph_poc.pt --output models/model_tgn.pt --epochs 80

# ─── Voos futuros ─────────────────────────────────────────────────────────
python predict.py --graph data/graph.pt --model models/model.pt --model-type gat \
    --future-since 2026-04-01 --output results/predictions.csv

# ─── Update incremental ───────────────────────────────────────────────────
python update_graph.py --since 2026-04-01 --graph data/graph.pt --output data/graph.pt
python train.py --model gat --graph data/graph.pt --finetune models/model.pt --output models/model.pt --epochs 20
```
