import json
import os
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

# === Argument parsing ===
parser = argparse.ArgumentParser(description="Evaluate model predictions per response file.")
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
    help="Path to folder containing response files (e.g. responses_1.jsonl … responses_n.jsonl)"
)
parser.add_argument(
    "--num_files",
    type=int,
    default=5,
    help="Number of response files (default: 5)"
)
args = parser.parse_args()

# === Process each file individually ===
for i in range(1, args.num_files + 1):
    file_path = os.path.join(args.base_path, f"responses_{i}.jsonl")
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        continue

    preds, labels = [], []
    with open(file_path, "r", encoding="utf‑8") as f:
        for line in f:
            data = json.loads(line)
            pred = data.get("model_response", "").strip().lower()
            label = data.get("query_stance", "").strip().lower()
            if pred in ["yes", "no"] and label in ["yes", "no"]:
                preds.append(pred)
                labels.append(label)

    print(f"\n=== File responses_{i}.jsonl ===")
    if not preds or not labels:
        print("No valid 'yes'/'no' predictions or labels found.")
        continue

    acc = accuracy_score(labels, preds)
    conf = confusion_matrix(labels, preds, labels=["yes", "no"])
    macro_f1 = f1_score(labels, preds, average="macro")
    report = classification_report(labels, preds, labels=["yes", "no"])

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"🎯 Macro‑F1: {macro_f1 * 100:.1f}%")
    print("\nConfusion Matrix:")
    print(conf)
    print("\nClassification Report:")
    print(report)