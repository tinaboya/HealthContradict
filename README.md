# HealthContradict

HealthContradict is a dataset for assessing how language models use contextual information to answer health questions, especially in the presence of conflicting contexts.

[Link to the paper.]

## Project Structure

```
HealthContradict/
├── scripts/                # All main Python scripts
│   ├── a1_pair_contradict_doc.py
│   ├── a2_create_dataset.py
│   ├── a3_create_prompt.py
│   ├── b1_inference.py
│   └── b2_eval_acc.py
├── dataset/                # Processed datasets and generated prompts
│   ├── dataset_ready.jsonl
│   ├── prompts_1.jsonl ... prompts_5.jsonl
├── doc/                    # Source document CSVs
│   ├── doc-all-stance.csv
│   └── doc-contradict.csv
├── query/                  # Query JSONL files
│   └── query-all.jsonl
```

## Setup

1. Install Python 3.12 and pip.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Script Usage

All scripts are in the `scripts/` folder. Run them from the project root:

### Model Inference
```bash
python3 scripts/b1_inference.py --model_path <MODEL_DIR> --prompt_dir dataset
```
Runs inference using a language model on the generated prompts.

### Evaluate Accuracy
```bash
python3 scripts/b2_eval_acc.py --base_path <RESPONSES_DIR> --num_files 5
```
Evaluates model predictions using accuracy and F1 metrics.

## Data Description
- `doc/`: Source document CSVs.
- `query/`: Queries JSONLs.
- `dataset/`: The dataset and the prompt files.

## Workflow
1. Run `a1_pair_contradict_doc.py` and `a2_create_dataset.py` to build the dataset.
3. Run `a3_create_prompt.py` to generate prompts.
4. Run `b1_inference.py` to get model responses.
5. Run `b2_eval_acc.py` to evaluate results.
