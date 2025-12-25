import torch
import torch.nn as nn
import torch.nn.functional as F
##########################################
# Unified Attention Gate (UAG)
##########################################

class UAG(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wg = nn.Linear(emb_dim, 1)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        vectors: [N, emb_dim]
        returns: [emb_dim]
        """
        if vectors.dim() != 2:
            raise ValueError(f"vectors must be 2D [N, emb_dim], got {vectors.shape}")

        # Self-attention
        Q = self.Wq(vectors)                     # [N, emb_dim]
        K = self.Wk(vectors)                     # [N, emb_dim]
        scores = torch.matmul(Q, K.T) / (self.emb_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, vectors)

        # Gated aggregation
        agg_logits = self.Wg(context)            # [N, 1]
        agg_weights = F.softmax(
            agg_logits.squeeze(-1), dim=0
        ).unsqueeze(-1)

        fused = torch.sum(agg_weights * context, dim=0)
        return fused

##########################################
# Latent Map
##########################################

class LatentMap(nn.Module):
    def __init__(self, input_dim: int, target_dim: int = 4):
        super().__init__()
        self.map = nn.Linear(input_dim, target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.map(x)


##########################################
# Temporal Fusion Module
##########################################

class TemporalFusion(nn.Module):
    def __init__(self, emb_dim=4, hidden_dim=4, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.uag = UAG(hidden_dim)

    def forward(self, visit_vectors):
        """
        visit_vectors: [num_visits, emb_dim]
        returns: [hidden_dim]
        """
        visit_seq = visit_vectors.unsqueeze(0)
        lstm_out, _ = self.lstm(visit_seq)
        lstm_out = lstm_out.squeeze(0)

        return self.uag(lstm_out)
##########################################
# Hierarchical Medical Fusion (HMF)
##########################################

class HMFModule(nn.Module):
    def __init__(self, emb_dim=4, hidden_dim=4):
        super().__init__()
        self.uag = UAG(emb_dim)
        self.latent_map_doc = LatentMap(emb_dim, emb_dim)
        self.latent_map_ts = LatentMap(emb_dim, emb_dim)
        self.temporal_fusion = TemporalFusion(emb_dim, hidden_dim)

    def forward(self, visits):
        """
        visits: list of dicts with keys:
            'meds'      : list of [emb_dim]
            'doctor'   : [emb_dim]
            'timestamp': [emb_dim]
        returns: [hidden_dim]
        """
        visit_vectors = []

        for visit in visits:
            meds = torch.stack(visit["meds"])
            med_vector = self.uag(meds)

            doc_proj = self.latent_map_doc(visit["doctor"])
            ts_proj  = self.latent_map_ts(visit["timestamp"])

            fused_modalities = torch.stack([med_vector, doc_proj, ts_proj])
            visit_vector = self.uag(fused_modalities)

            visit_vectors.append(visit_vector)

        visit_seq = torch.stack(visit_vectors)
        return self.temporal_fusion(visit_seq)
##########################################
# Interval Modeling
##########################################

def compute_intervals(timestamps):
    timestamps = torch.tensor(timestamps, dtype=torch.float)
    return torch.diff(timestamps, prepend=timestamps[:1])


class IntervalEncoder(nn.Module):
    def __init__(self, num_bins=10, emb_dim=4):
        super().__init__()
        self.embed = nn.Embedding(num_bins, emb_dim)

    def forward(self, intervals):
        bins = torch.clamp(
            (intervals // 7).long(),
            max=self.embed.num_embeddings - 1
        )
        return self.embed(bins)


class IntervalLSTM(nn.Module):
    def __init__(self, emb_dim=4, hidden_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, interval_embeddings):
        out, _ = self.lstm(interval_embeddings.unsqueeze(0))
        return out.squeeze(0)


class IntervalModule(nn.Module):
    def __init__(self, num_bins=10, emb_dim=4, hidden_dim=4):
        super().__init__()
        self.interval_encoder = IntervalEncoder(num_bins, emb_dim)
        self.lstm_encoder = IntervalLSTM(emb_dim, hidden_dim)
        self.uag = UAG(hidden_dim)

    def forward(self, timestamps):
        intervals = compute_intervals(timestamps)
        interval_embeds = self.interval_encoder(intervals)
        lstm_out = self.lstm_encoder(interval_embeds)
        return self.uag(lstm_out)
##########################################
# Graph Structure Modules
##########################################

class MetaPathGNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, node_features, adjacency_matrix):
        aggregated = torch.matmul(adjacency_matrix, node_features)
        return self.linear(aggregated)


class StructuralLSTM(nn.Module):
    def __init__(self, emb_dim=4, hidden_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, structural_sequence):
        out, _ = self.lstm(structural_sequence.unsqueeze(0))
        return out.squeeze(0)


class GraphStructureModule(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=4):
        super().__init__()
        self.gnn = MetaPathGNN(input_dim, input_dim)
        self.lstm = StructuralLSTM(input_dim, hidden_dim)
        self.uag = UAG(hidden_dim)

    def forward(self, claimer_node_id, graph_snapshots):
        structural_sequence = []

        for node_features, adj in graph_snapshots:
            gnn_out = self.gnn(node_features, adj)
            structural_sequence.append(gnn_out[claimer_node_id])

        structural_sequence = torch.stack(structural_sequence)
        lstm_out = self.lstm(structural_sequence)
        return self.uag(lstm_out)
##########################################
# Final Fraud Detection Model
##########################################

class FraudDetectionModel(nn.Module):
    def __init__(self, emb_dim=4, hidden_dim=4, num_classes=2):
        super().__init__()
        self.hmf = HMFModule(emb_dim, hidden_dim)
        self.interval = IntervalModule(10, emb_dim, hidden_dim)
        self.graph = GraphStructureModule(emb_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, visits, timestamps, claimer_node_id, graph_snapshots):
        E_hmf   = self.hmf(visits)
        E_int   = self.interval(timestamps)
        E_graph = self.graph(claimer_node_id, graph_snapshots)

        fused = torch.cat([E_hmf, E_int, E_graph], dim=-1)
        return self.classifier(fused)
