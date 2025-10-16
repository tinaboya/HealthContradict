import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import re

# === Argument parser ===
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory.")
parser.add_argument("--prompt_dir", type=str, default="dataset", help="Directory containing prompt files.")
args = parser.parse_args()

# === Load model and tokenizer once ===
print(f"🔄 Loading model from {args.model_path}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Model loaded on device: {device}")

# === Function for predict mode ===
def predict_next_token_label(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]  # logits for next token

        label_tokens = ["YES", "NO"]
        label_ids = tokenizer.convert_tokens_to_ids(label_tokens)

        selected_logits = next_token_logits[label_ids]  # logits for 'YES' and 'NO'
        probs = torch.softmax(selected_logits, dim=-1)

        output_prob = {label: probs[i].item() for i, label in enumerate(label_tokens)}
        output_logits = {label: selected_logits[i].item() for i, label in enumerate(label_tokens)}

        best_label = max(output_prob, key=output_prob.get)

        return best_label, output_prob, output_logits

# === Loop through all prompt files ===
model_name = os.path.basename(os.path.normpath(args.model_path))
output_dir = os.path.join("results", model_name)
os.makedirs(output_dir, exist_ok=True)

prompt_files = sorted([
    f for f in os.listdir(args.prompt_dir)
    if re.match(r"prompts_\d+\.jsonl$", f)
    and 5 <= int(re.search(r"\d+", f).group()) <= 5
])

for prompt_file in prompt_files:
    input_path = os.path.join(args.prompt_dir, prompt_file)
    index = re.search(r'\d+', prompt_file).group()
    output_path = os.path.join(output_dir, f"responses_{index}.jsonl")

    print(f"📝 Processing {prompt_file} → {output_path}")
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc=f"Generating for prompts_{index}"):
            data = json.loads(line)
            prompt = data["prompt"]
            response, softmax_prob, logits = predict_next_token_label(model, tokenizer, prompt, device)

            output_entry = {
                "instance_id": data["instance_id"],
                "template_id": data["template_id"],
                "prompt": prompt,
                "model_response": response,
                "query_stance": data["query_stance"]
            }

            if softmax_prob:
              output_entry["softmax_prob"] = softmax_prob
            if logits:
              output_entry["logits"] = logits

            f_out.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

    print(f"✅ Done: {output_path}")