import torch
import torch.nn as nn
from components import FraudDetectionModel


def train_model(train_data, model: nn.Module, num_epochs: int = 10, lr: float = 0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0

        for sample in train_data:
            visits = sample["visits"]
            timestamps = sample["timestamps"]
            claimer_node_id = sample["claimer_node_id"]
            graph_snapshots = sample["graph_snapshots"]
            label = int(sample["label"])

            optimizer.zero_grad()
            logits = model(visits, timestamps, claimer_node_id, graph_snapshots)

            # Ensure target is 1D long tensor matching batch dimension
            target = torch.tensor([label], dtype=torch.long, device=logits.device)
            loss = criterion(logits.unsqueeze(0), target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(logits).item()
            correct += int(pred == label)

        acc = correct / max(1, len(train_data))
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {acc:.2%}")


if __name__ == '__main__':
    # Quick smoke test with dummy data to validate shapes and forward pass
    emb_dim = 4
    hidden_dim = 4
    model = FraudDetectionModel(emb_dim=emb_dim, hidden_dim=hidden_dim, num_classes=2)

    # Build one dummy visit
    meds = [torch.randn(emb_dim) for _ in range(3)]
    visit = {
        "meds": meds,
        "doctor": torch.randn(emb_dim),
        "timestamp": torch.randn(emb_dim),
    }

    visits = [visit, visit]  # two visits

    timestamps = [0, 7]

    # Graph snapshots: two time steps, each with 1 node of emb_dim
    node_features = torch.randn(1, emb_dim)
    adj = torch.eye(1)
    graph_snapshots = [(node_features, adj), (node_features, adj)]

    logits = model(visits, timestamps, claimer_node_id=0, graph_snapshots=graph_snapshots)
    print('Logits:', logits)