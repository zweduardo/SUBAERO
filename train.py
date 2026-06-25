# train.py — Treina GNN heterogenea para predicao de atraso de chegada
#
# Entrada : graph.pt (gerado por build_graph.py)
# Saida   : model.pt (pesos), results.csv (metricas por epoca)
#
# Arquiteturas disponiveis (--model):
#   gat : HeteroConv(GATConv) x 2 camadas
#   hgt : HGTConv x 2 camadas (Heterogeneous Graph Transformer)
#   tgn : Temporal-aware GNN com memoria GRU ao longo de NEXT_LEG
#
# Uso:
#   python train.py --model gat --graph graph_poc.pt --epochs 80
#   python train.py --model hgt --graph graph_poc.pt --epochs 80
#   python train.py --model tgn --graph graph_poc.pt --epochs 80

import argparse
import math
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, HGTConv, Linear
from torch_geometric.transforms import ToUndirected
from sklearn.model_selection import train_test_split
import numpy as np


# ======================================================================
#  Modelo 1: GAT (Graph Attention Network heterogenea)
# ======================================================================

class FlightDelayGAT(nn.Module):
    """
    HeteroConv com GATConv por tipo de aresta.
    Cada relacao tem pesos independentes de atencao.
    """

    def __init__(self, metadata, in_channels_dict, hidden=64, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        self.conv1 = HeteroConv(
            {
                etype: GATConv(
                    in_channels=(-1, -1),
                    out_channels=hidden,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=True,
                )
                for etype in metadata[1]
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                etype: GATConv(
                    in_channels=(-1, -1),
                    out_channels=hidden,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=False,
                )
                for etype in metadata[1]
            },
            aggr="sum",
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                  for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}

        out = self.head(x_dict["flight"]).squeeze(-1)
        return out


# ======================================================================
#  Modelo 2: HGT (Heterogeneous Graph Transformer)
# ======================================================================

class FlightDelayHGT(nn.Module):
    """
    Heterogeneous Graph Transformer — cada tipo de no e aresta tem
    transformacoes especificas com atencao multi-head.
    """

    def __init__(self, metadata, in_channels_dict, hidden=64, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # HGTConv exige que todos os nos tenham a mesma dimensao de entrada
        # -> projecao linear por tipo de no
        self.projections = nn.ModuleDict()
        for ntype, in_dim in in_channels_dict.items():
            self.projections[ntype] = nn.Linear(in_dim, hidden)

        self.conv1 = HGTConv(
            in_channels=hidden,
            out_channels=hidden,
            metadata=metadata,
            heads=heads,
        )

        self.conv2 = HGTConv(
            in_channels=hidden,
            out_channels=hidden,
            metadata=metadata,
            heads=heads,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        # Projecao para dimensao comum
        x_dict = {k: F.relu(self.projections[k](v)) for k, v in x_dict.items()}

        # Camada 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                  for k, v in x_dict.items()}

        # Camada 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}

        out = self.head(x_dict["flight"]).squeeze(-1)
        return out


# ======================================================================
#  Modelo 3: TGN-like (Temporal Graph Network adaptado)
# ======================================================================

class FlightDelayTGN(nn.Module):
    """
    GNN com memoria temporal inspirada no TGN.

    O TGN original opera sobre grafos de interacao temporal (TemporalData).
    Aqui adaptamos a ideia central para nosso grafo heterogeneo:

    1. Projecao linear por tipo de no
    2. Memoria GRU ao longo das cadeias NEXT_LEG — simula o modulo de
       memoria do TGN, propagando estado temporal de voo em voo na
       mesma aeronave (onde o atraso realmente "viaja")
    3. HeteroConv(GATConv) para agregar contexto espacial (aeroporto,
       clima, aeronave)
    4. Cabeca de predicao

    Isso captura as duas ideias-chave do TGN:
    - Memoria por no que evolui no tempo (GRU sobre NEXT_LEG)
    - Agregacao de vizinhanca (graph attention)
    """

    def __init__(self, metadata, in_channels_dict, hidden=64, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.hidden = hidden

        # Projecao por tipo de no
        self.projections = nn.ModuleDict()
        for ntype, in_dim in in_channels_dict.items():
            self.projections[ntype] = nn.Linear(in_dim, hidden)

        # Memoria temporal: GRU que processa ao longo de NEXT_LEG
        self.memory_gru = nn.GRUCell(hidden, hidden)

        # Convolucao espacial (agrega clima, aeroporto, aeronave)
        self.conv = HeteroConv(
            {
                etype: GATConv(
                    in_channels=(-1, -1),
                    out_channels=hidden,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=False,
                )
                for etype in metadata[1]
            },
            aggr="sum",
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        # Cache da ordem topologica (calculada uma vez)
        self._topo_order = None

    def _compute_topo_order(self, edge_index_dict, n_flights):
        """
        Computa ordem topologica das cadeias NEXT_LEG.
        Retorna lista de indices de voos em ordem temporal.
        """
        # Procura a aresta NEXT_LEG (pode ser original ou reversa)
        next_leg_key = None
        for key in edge_index_dict:
            if "NEXT_LEG" in key[1]:
                src_type, rel, dst_type = key
                if src_type == "flight" and dst_type == "flight":
                    next_leg_key = key
                    break

        if next_leg_key is None or edge_index_dict[next_leg_key].shape[1] == 0:
            # Sem NEXT_LEG, retorna ordem natural
            return list(range(n_flights))

        ei = edge_index_dict[next_leg_key]
        src, dst = ei[0].cpu().numpy(), ei[1].cpu().numpy()

        # Encontra raizes (nos sem predecessores no NEXT_LEG)
        has_predecessor = set(dst)
        roots = [i for i in range(n_flights) if i not in has_predecessor]

        # Monta grafo de adjacencia
        adj = {}
        for s, d in zip(src, dst):
            adj.setdefault(int(s), []).append(int(d))

        # BFS a partir das raizes para ordem topologica
        visited = set()
        order = []
        queue = list(roots)
        for r in queue:
            visited.add(r)
        idx = 0
        while idx < len(queue):
            node = queue[idx]
            idx += 1
            order.append(node)
            for nxt in adj.get(node, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        # Nos isolados (sem NEXT_LEG)
        for i in range(n_flights):
            if i not in visited:
                order.append(i)

        return order

    def forward(self, x_dict, edge_index_dict):
        # 1. Projecao
        x_dict = {k: F.relu(self.projections[k](v)) for k, v in x_dict.items()}

        # 2. Memoria temporal via GRU ao longo de NEXT_LEG
        flight_h = x_dict["flight"]  # [N, hidden]
        n_flights = flight_h.shape[0]
        device = flight_h.device

        # Computa ordem topologica (cache)
        if self._topo_order is None:
            self._topo_order = self._compute_topo_order(edge_index_dict, n_flights)

        # Monta mapa de predecessores NEXT_LEG
        next_leg_key = None
        for key in edge_index_dict:
            if "NEXT_LEG" in key[1] and key[0] == "flight" and key[2] == "flight":
                next_leg_key = key
                break

        predecessor = {}
        if next_leg_key is not None and edge_index_dict[next_leg_key].shape[1] > 0:
            ei = edge_index_dict[next_leg_key]
            src, dst = ei[0], ei[1]
            for s, d in zip(src.tolist(), dst.tolist()):
                predecessor[d] = s

        # Propaga memoria na ordem topologica (sem operacoes in-place)
        zero_h = torch.zeros(1, self.hidden, device=device)
        mem_list = [None] * n_flights
        for node in self._topo_order:
            pred = predecessor.get(node)
            h_prev = mem_list[pred] if pred is not None else zero_h
            mem_list[node] = self.memory_gru(
                flight_h[node].unsqueeze(0), h_prev
            )  # [1, hidden]

        memory = torch.cat(mem_list, dim=0)  # [N, hidden]

        # Combina features projetadas com memoria temporal
        x_dict["flight"] = flight_h + memory  # residual connection

        # 3. Convolucao espacial
        x_dict = self.conv(x_dict, edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}

        # 4. Predicao
        out = self.head(x_dict["flight"]).squeeze(-1)
        return out


# ======================================================================
#  Factory
# ======================================================================

MODEL_REGISTRY = {
    "gat": FlightDelayGAT,
    "hgt": FlightDelayHGT,
    "tgn": FlightDelayTGN,
}


def build_model(name, metadata, in_channels_dict, hidden, heads, dropout):
    cls = MODEL_REGISTRY[name]
    return cls(
        metadata=metadata,
        in_channels_dict=in_channels_dict,
        hidden=hidden,
        heads=heads,
        dropout=dropout,
    )


# ======================================================================
#  Splits de treino / val / teste
# ======================================================================

def make_masks(n, val_ratio=0.15, test_ratio=0.15, seed=42):
    idx = np.arange(n)
    idx_train, idx_tmp = train_test_split(
        idx, test_size=val_ratio + test_ratio, random_state=seed
    )
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
    )

    def to_mask(indices):
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.tensor(indices, dtype=torch.long)] = True
        return m

    return to_mask(idx_train), to_mask(idx_val), to_mask(idx_test)


# ======================================================================
#  Loop de treino
# ======================================================================

def train_epoch(model, data, optimizer, mask):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x_dict, data.edge_index_dict)
    loss = F.mse_loss(pred[mask], data["flight"].y[mask])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    """Retorna (MSE_normalizado, MAE_minutos, RMSE_minutos)."""
    model.eval()
    pred_norm = model(data.x_dict, data.edge_index_dict)[mask]
    y_norm    = data["flight"].y[mask]

    mse_norm = F.mse_loss(pred_norm, y_norm).item()

    y_mean = data["flight"].y_mean
    y_std  = data["flight"].y_std
    pred_min = pred_norm * y_std + y_mean
    y_min    = y_norm    * y_std + y_mean

    mae  = (pred_min - y_min).abs().mean().item()
    rmse = math.sqrt(((pred_min - y_min) ** 2).mean().item())
    return mse_norm, mae, rmse


# ======================================================================
#  Main
# ======================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    print(f"Modelo: {args.model.upper()}")

    # -- Carregar grafo
    print(f"\nCarregando '{args.graph}'...")
    data: HeteroData = torch.load(args.graph, weights_only=False)

    data = data.to(device)

    # Compatibilidade com grafos buildados antes da v2 do build_graph.py
    edge_names = [et[1] for et in data.edge_types]
    if not any(n.startswith("rev_") for n in edge_names):
        print("  Aplicando ToUndirected (grafo sem arestas reversas)...")
        data = ToUndirected()(data)

    n_flights = data["flight"].x.shape[0]
    print(f"  {n_flights} voos  |  {len(data.edge_types)} tipos de aresta")

    # -- Masks
    if args.finetune and hasattr(data["flight"], "new_mask"):
        # Modo fine-tune: treina so nos voos novos (marcados por update_graph.py)
        new_mask = data["flight"].new_mask.to(device)
        n_new = new_mask.sum().item()
        print(f"  Modo FINE-TUNE: {n_new} voos novos detectados")

        # Split dos novos em treino (85%) e val (15%)
        new_indices = new_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        from sklearn.model_selection import train_test_split as tts
        idx_train, idx_val = tts(new_indices, test_size=0.15, random_state=42)

        train_mask = torch.zeros(n_flights, dtype=torch.bool)
        val_mask = torch.zeros(n_flights, dtype=torch.bool)
        test_mask = val_mask.clone()  # val = test no fine-tune
        train_mask[torch.tensor(idx_train, dtype=torch.long)] = True
        val_mask[torch.tensor(idx_val, dtype=torch.long)] = True
        test_mask = val_mask.clone()
    else:
        train_mask, val_mask, test_mask = make_masks(n_flights)

    data["flight"].train_mask = train_mask.to(device)
    data["flight"].val_mask   = val_mask.to(device)
    data["flight"].test_mask  = test_mask.to(device)

    print(f"  Split - treino: {train_mask.sum()}  val: {val_mask.sum()}  "
          f"teste: {test_mask.sum()}")

    # -- Modelo
    in_channels_dict = {nt: data[nt].x.shape[1] for nt in data.node_types}

    model = build_model(
        name=args.model,
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        hidden=args.hidden,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    # Forward pass dummy para inicializar parametros lazy (GAT usa -1,-1)
    with torch.no_grad():
        model.eval()
        _ = model(data.x_dict, data.edge_index_dict)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametros treinaveis: {n_params:,}")

    # Carregar pesos existentes no modo fine-tune
    if args.finetune:
        print(f"  Carregando pesos de '{args.finetune}'...")
        model.load_state_dict(torch.load(args.finetune, map_location=device, weights_only=True))
        print(f"  Pesos carregados. Fine-tuning com lr={args.lr}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # -- Treino
    print(f"\nTreinando por {args.epochs} epocas...\n")
    header = f"{'Epoca':>6}  {'Loss':>9}  {'Val MAE':>9}  {'Val RMSE':>10}  {'LR':>10}"
    print(header)
    print("-" * len(header))

    best_val_mae = float("inf")
    best_epoch   = 0
    results      = []

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, data, optimizer, data["flight"].train_mask)
        _, val_mae, val_rmse = evaluate(model, data, data["flight"].val_mask)
        scheduler.step(val_mae)

        lr_now = optimizer.param_groups[0]["lr"]
        results.append([epoch, loss, val_mae, val_rmse, lr_now])

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"{epoch:>6}  {loss:>9.4f}  {val_mae:>9.2f}  "
                  f"{val_rmse:>10.2f}  {lr_now:>10.2e}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch   = epoch
            torch.save(model.state_dict(), args.output)

    # -- Avaliacao final
    print(f"\nMelhor modelo: epoca {best_epoch}  val MAE = {best_val_mae:.2f} min")
    model.load_state_dict(torch.load(args.output, weights_only=True))

    _, test_mae, test_rmse = evaluate(model, data, data["flight"].test_mask)
    print(f"\n{'-'*40}")
    print(f"  [{args.model.upper()}] Teste - MAE  : {test_mae:.2f} min")
    print(f"  [{args.model.upper()}] Teste - RMSE : {test_rmse:.2f} min")
    print(f"{'-'*40}\n")

    # -- Salva metricas CSV
    csv_path = Path(args.output).with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_mae", "val_rmse", "lr"])
        writer.writerows(results)
    print(f"Metricas salvas em '{csv_path}'")
    print(f"Modelo salvo em '{args.output}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina GNN para predicao de atraso")
    parser.add_argument("--model",     default="gat", choices=["gat", "hgt", "tgn"],
                        help="Arquitetura: gat, hgt ou tgn")
    parser.add_argument("--graph",     default="graph.pt",  help="Grafo de entrada")
    parser.add_argument("--output",    default="model.pt",  help="Pesos do modelo")
    parser.add_argument("--finetune",  default=None,
                        help="Caminho para pesos existentes (ativa modo fine-tune)")
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--wd",        type=float, default=1e-4,  help="Weight decay")
    parser.add_argument("--hidden",    type=int,   default=64,    help="Dim oculta")
    parser.add_argument("--heads",     type=int,   default=4,     help="Cabecas de atencao")
    parser.add_argument("--dropout",   type=float, default=0.3)
    parser.add_argument("--log-every", type=int,   default=5,     dest="log_every")
    args = parser.parse_args()
    main(args)
