import pandas as pd
import json

# === Config ===
RANDOM_SEED = 42
docs_csv = '../doc/doc-all-stance.csv'
cleaned_query_file = '../query/query-all.jsonl'
output_pairs_csv = '../doc/doc-contradict.csv'

# === Step 1: Load cleaned queries and get valid topic_ids ===
valid_topic_ids = set()
with open(cleaned_query_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        valid_topic_ids.add(int(data["topic_id"]))

# === Step 2: Load stance-labeled documents ===
df = pd.read_csv(docs_csv)

# === Step 3: Define pairing function ===
def generate_unique_contradiction_pairs(topic_df, topic_id, seed, used_doc_ids):
    topic_df = topic_df[topic_df['text'].apply(lambda x: isinstance(x, str) and x.strip() != '')]
    topic_df = topic_df[~topic_df['doc_id'].isin(used_doc_ids)]

    yes_docs = topic_df[topic_df['stance'] == 'yes'].sample(frac=1, random_state=seed).reset_index(drop=True)
    no_docs = topic_df[topic_df['stance'] == 'no'].sample(frac=1, random_state=seed + 1).reset_index(drop=True)

    pair_count = min(len(yes_docs), len(no_docs))
    contradiction_pairs = []

    for i in range(pair_count):
        yes_id = yes_docs.loc[i, 'doc_id']
        no_id = no_docs.loc[i, 'doc_id']
        contradiction_pairs.append((topic_id, yes_id, no_id, 'contradiction'))

        used_doc_ids.update([yes_id, no_id])

    return contradiction_pairs

# === Step 4: Generate contradiction pairs only for valid topics ===
used_doc_ids = set()
contradiction_pairs = []

for topic_id, topic_df in df.groupby('topic_id'):
    if topic_id not in valid_topic_ids:
        continue

    seed = RANDOM_SEED + int(topic_id)
    pairs = generate_unique_contradiction_pairs(topic_df, topic_id, seed, used_doc_ids)
    contradiction_pairs.extend(pairs)

# === Step 5: Save results ===
result_df = pd.DataFrame(contradiction_pairs, columns=['topic_id', 'doc1_id', 'doc2_id', 'pair_type'])
result_df.to_csv(output_pairs_csv, index=False)

print(f"✅ Total contradiction pairs created: {len(result_df)}")
print(f"✅ Total unique documents used: {len(used_doc_ids)}")
