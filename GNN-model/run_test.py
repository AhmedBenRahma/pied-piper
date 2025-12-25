from components import FraudDetectionModel
from latentmapmaker import train_model
import torch
import random

def generate_synthetic_data(num_samples=100, emb_dim=4, max_visits=5, max_meds=4, num_nodes=5):
    data = []

    for _ in range(num_samples):
        num_visits = random.randint(1, max_visits)
        visits = []
        timestamps = []
        graph_snapshots = []

        current_time = 0
        for _ in range(num_visits):
            num_meds = random.randint(1, max_meds)
            meds = [torch.randn(emb_dim) for _ in range(num_meds)]
            doctor = torch.randn(emb_dim)
            timestamp = torch.randn(emb_dim)

            visits.append({
                "meds": meds,
                "doctor": doctor,
                "timestamp": timestamp
            })

            timestamps.append(current_time)
            current_time += random.randint(1, 10)  # simulate time gaps

            # Graph snapshot for this time step
            node_features = torch.randn(num_nodes, emb_dim)
            adjacency_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
            graph_snapshots.append((node_features, adjacency_matrix))

        label = random.randint(0, 1)  # 0 = not fraud, 1 = fraud

        sample = {
            "visits": visits,
            "timestamps": timestamps,
            "claimer_node_id": 0,
            "graph_snapshots": graph_snapshots,
            "label": label
        }

        data.append(sample)

    return data

# Generate training and testing datasets

test_data = generate_synthetic_data(num_samples=300)

def test_model(test_data, model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for sample in test_data:
            visits = sample["visits"]
            timestamps = sample["timestamps"]
            claimer_node_id = sample["claimer_node_id"]
            graph_snapshots = sample["graph_snapshots"]
            label = sample["label"]

            logits = model(visits, timestamps, claimer_node_id, graph_snapshots)
            pred = torch.argmax(logits).item()
            correct += int(pred == label)

    acc = correct / max(1, len(test_data))
    print(f"Test Accuracy: {acc:.2%}")


if __name__ == '__main__':
    train_data = generate_synthetic_data(num_samples=300)
    model = FraudDetectionModel(emb_dim=4, hidden_dim=4)

    # Train briefly (small epochs for smoke test)
    train_model(train_data, model, num_epochs=100 , lr=0.1)

    # Test on same data
    test_model(train_data, model)
