import torch
import random
import pandas as pd

def generate_synthetic_data(num_samples=100, emb_dim=4, max_visits=5, max_meds=4, num_nodes=5):
    data = []

    for sample_id in range(num_samples):
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
            current_time += random.randint(1, 10)

            node_features = torch.randn(num_nodes, emb_dim)
            adjacency_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
            graph_snapshots.append((node_features, adjacency_matrix))

        label = random.randint(0, 1)

        data.append({
            "sample_id": sample_id,
            "visits": visits,
            "timestamps": timestamps,
            "claimer_node_id": 0,
            "graph_snapshots": graph_snapshots,
            "label": label
        })

    return data

def flatten_data_for_csv(data, emb_dim=4):
    rows = []
    for sample in data:
        sample_id = sample["sample_id"]
        label = sample["label"]
        for i, visit in enumerate(sample["visits"]):
            row = {
                "sample_id": sample_id,
                "visit_index": i,
                "label": label,
                "timestamp_value": sample["timestamps"][i]
            }
            # Flatten doctor and timestamp embeddings
            for j in range(emb_dim):
                row[f"doctor_{j}"] = visit["doctor"][j].item()
                row[f"timestamp_{j}"] = visit["timestamp"][j].item()
            # Flatten meds
            for m, med in enumerate(visit["meds"]):
                for j in range(emb_dim):
                    row[f"med_{m}_{j}"] = med[j].item()
            rows.append(row)
    return pd.DataFrame(rows)

# Generate and save
train_data = generate_synthetic_data(num_samples=200)
test_data = generate_synthetic_data(num_samples=50)

train_df = flatten_data_for_csv(train_data)
test_df = flatten_data_for_csv(test_data)

# Save to CSV
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("✅ train_data.csv and test_data.csv saved to your working directory.")