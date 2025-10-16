import pandas as pd
import json

# === File paths ===
pairs_csv = "../doc/doc-contradict.csv"
docs_csv = "../doc/doc-all-stance.csv"
query_file = "../query/query-all.jsonl"
output_jsonl = "../dataset/dataset_ready.jsonl"

# === Step 1: Load cleaned queries ===
query_lookup = {}
with open(query_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        topic_id = int(data["topic_id"])
        query_lookup[topic_id] = {
            "query": data["description"],
            "query_stance": data["query_stance"]
        }

# === Step 2: Load documents and pairs ===
df_docs = pd.read_csv(docs_csv)
df_pairs = pd.read_csv(pairs_csv)

# Index documents by (topic_id, doc_id)
doc_lookup = {
    (row['topic_id'], row['doc_id']): {
        'stance': row['stance'],
        'text': row['text']
    }
    for _, row in df_docs.iterrows()
}

# === Step 3: Build and write final prompt-ready JSONL ===
with open(output_jsonl, "w", encoding="utf-8") as f_out:
    instance_id = 0
    for _, row in df_pairs.iterrows():
        topic_id = row['topic_id']
        doc1_id = row['doc1_id']
        doc2_id = row['doc2_id']

        doc1 = doc_lookup.get((topic_id, doc1_id))
        doc2 = doc_lookup.get((topic_id, doc2_id))
        query_data = query_lookup.get(topic_id)

        if not doc1 or not doc2 or not query_data:
            print(f"⚠️ Missing data for topic {topic_id} or documents {doc1_id}, {doc2_id}")
            continue

        # Map to YES and NO
        if doc1['stance'] == 'yes' and doc2['stance'] == 'no':
            doc_a, doc_b = doc1['text'], doc2['text']
        elif doc2['stance'] == 'yes' and doc1['stance'] == 'no':
            doc_a, doc_b = doc2['text'], doc1['text']
        else:
            print(f"⚠️ Skipping non-yes/no pair: {doc1_id}, {doc2_id}")
            continue

        entry = {
            "instance_id": instance_id,
            "topic_id": topic_id,
            "query": query_data["query"],
            "query_stance": query_data["query_stance"],
            "doc_a": doc_a,
            "doc_b": doc_b
        }

        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        instance_id += 1

print(f"✅ Saved prompt-ready data to: {output_jsonl}")
