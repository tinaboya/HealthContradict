# script: generate_prompts_by_template.py

import json
import os

# === File paths ===
input_jsonl = "../dataset/dataset_ready.jsonl"
output_dir = "../dataset"

os.makedirs(output_dir, exist_ok=True)

def build_prompt(template_id, query, doc_a, doc_b):
    if template_id == 1:
        return f"Answer the following question with only YES or NO based on your parametric knowledge.\nQuestion: {query} \nAnswer:"
    elif template_id == 2:
        return f"Answer the following question with only YES or NO based on the given contextual knowledge.\nQuestion: {query}\nContext: {correct_doc}  \nAnswer:"
    elif template_id == 3:
        return f"Answer the following question with only YES or NO based on the given contextual knowledge.\nQuestion: {query}\nContext: {incorrect_doc} \nAnswer:"
    elif template_id == 4:
        return f"Answer the following question with only YES or NO based on the given contextual knowledge.\nQuestion: {query}\nContext:\n{correct_doc}\n\n{incorrect_doc} \nAnswer:"
    elif template_id == 5:
        return f"Answer the following question with only YES or NO based on the given contextual knowledge.\nQuestion: {query}\nContext:\n{incorrect_doc}\n\n{correct_doc} \nAnswer:"
    else:
        raise ValueError(f"Invalid template_id: {template_id}")

# === Open 5 files ===
output_files = {
    template_id: open(os.path.join(output_dir, f"prompts_{template_id}.jsonl"), "w", encoding="utf-8")
    for template_id in range(1, 6)
}

# === Generate prompts per template ===
with open(input_jsonl, "r", encoding="utf-8") as f_in:
    for line in f_in:
        data = json.loads(line)
        instance_id = data["instance_id"]
        topic_id = data["topic_id"]
        query = data["query"]
        query_stance = data["query_stance"]
        doc_a = data["doc_a"]
        doc_b = data["doc_b"]

        # Determine correct and incorrect docs based on stance
        if query_stance == "yes":
            correct_doc = doc_a
            incorrect_doc = doc_b
        else:
            correct_doc = doc_b
            incorrect_doc = doc_a

        for template_id in range(1, 6):
            prompt_text = build_prompt(template_id, query, doc_a, doc_b)

            entry = {
                "instance_id": instance_id,
                "topic_id": topic_id,
                "template_id": template_id,
                "query": query,
                "query_stance": query_stance,
                "prompt": prompt_text
            }

            output_files[template_id].write(json.dumps(entry, ensure_ascii=False) + "\n")

# === Close all files ===
for f in output_files.values():
    f.close()

print("✅ Saved prompts to prompts_template1–5.jsonl")
