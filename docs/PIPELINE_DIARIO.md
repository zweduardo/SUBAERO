# Pipeline Diário — TCC GNN Flight Delay

## Visão Geral

O pipeline diário atualiza o modelo HGT com novos dados de voos sem precisar retreinar do zero.

```
02:00  load_daily.py              → carrega voos do dia no Neo4j
03:00  build_train_iterative_rotation_daily.py --finetune  → partial fit no HGT
```

---

## Scripts Principais

| Script | Papel |
|--------|-------|
| `load_daily.py` | Carrega voos de um dia via VRA API → Neo4j |
| `build_train_iterative_rotation_daily.py` | Treino iterativo por aeroporto (GAT/HGT) com suporte a partial fit e tracking W&B |
| `load_data.py` | Carga histórica completa (2020-2024) |
| `api_calls.py` | Wrappers de APIs externas (VRA, Open-Meteo, OpenSky) |

---

## Partial Fit

### Como funciona

O script carrega um modelo existente (`--finetune`) e faz algumas épocas de fine-tuning com os períodos recentes (`--since`). O LR é reduzido automaticamente 100x para evitar catastrofic forgetting.

### Mudanças feitas no script original

1. **Checkpoint completo** — `torch.save` agora salva além dos pesos: `model_type`, `hidden`, `heads`, `dropout`, `metadata`, `in_channels_dict`. Necessário para reconstruir o HGT sem repassar os args.

2. **`load_model(path)`** — nova função que reconstrói o modelo do zero a partir do checkpoint.

3. **`--finetune <path>`** — carrega modelo existente em vez de inicializar do zero.

4. **`--since YYYY-MM-DD`** — filtra apenas períodos a partir dessa data.

5. **LR automático** — `lr * 0.01` quando `--finetune` está ativo.

6. **W&B tracking** — cada run (treino inicial ou partial fit) é logada como run separada no projeto `tcc-flight-delay`.

### Uso

```bash
# Treino inicial (primeira vez)
python -u build_train_iterative_rotation_daily.py --model hgt --epochs 50 --output model_iterative_rotation.pt

# Partial fit diário
python -u build_train_iterative_rotation_daily.py \
  --model hgt \
  --finetune model_iterative_rotation.pt \
  --since 2024-12-31 \
  --epochs 5 \
  --output model_iterative_rotation.pt
```

---

## Raspberry Pi

### Localização
- IP: `192.168.15.118`
- Projeto: `/home/admin/tcc/`
- Python (venv): `/home/admin/tcc/venv/bin/python3`
- Logs: `/home/admin/tcc/logs/`

### Setup inicial (já executado)

```bash
# Arquivos transferidos via SFTP:
# - build_train_iterative_rotation_daily.py
# - load_daily.py
# - load_data.py
# - api_calls.py
# - model_iterative_rotation.pt
# - credentials.json

# Venv criado e dependências instaladas:
python3 -m venv /home/admin/tcc/venv
/home/admin/tcc/venv/bin/pip install torch torch_geometric neo4j pandas numpy scikit-learn wandb requests
```

### W&B no Pi

Rodar uma vez manualmente após o setup:
```bash
/home/admin/tcc/venv/bin/wandb login
# Cola a API key de https://wandb.ai/authorize
```

### Cron Jobs

```
0 2 * * *  cd /home/admin/tcc && venv/bin/python3 -u load_daily.py >> logs/load_daily_$(date +%Y%m%d).log 2>&1
0 3 * * *  cd /home/admin/tcc && venv/bin/python3 -u build_train_iterative_rotation_daily.py --model hgt --finetune model_iterative_rotation.pt --since $(date -d 'yesterday' +%Y-%m-%d) --epochs 5 --output model_iterative_rotation.pt >> logs/partial_fit_$(date +%Y%m%d).log 2>&1
```

Configurar com:
```bash
bash ~/tcc/setup_cron.sh
```

---

## W&B — Métricas Rastreadas

| Métrica | Descrição |
|---------|-----------|
| `train_mse` | MSE no treino (escala z-score) |
| `val_mae` | MAE na validação em minutos — métrica principal |
| `test_mae` | MAE no teste em minutos (ao final do treino) |
| `lr` | Learning rate atual |

Cada run identifica se é treino inicial ou partial fit pelo nome: `train-hgt-YYYYMMDD-HHMM` ou `finetune-hgt-YYYYMMDD`.

Dashboard: https://wandb.ai/eduardozw-www-icmc-usp-br/tcc-flight-delay
