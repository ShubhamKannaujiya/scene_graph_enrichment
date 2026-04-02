# Scene Graph Enrichment Pipeline

This project enriches scene graph triplets with:
- Intent  
- Causal effect  
- Counterfactual  

using a multimodal LLM (Gemma 3).

---

## 📁 Folder Structure

project/
│
├── scene_graph_pipeline.py      # Main script
├── input/
│   └── cleaned_vg.jsonl        # Input data
├── imagery/
│   └── *.jpg                   # Images (optional)
├── output/
│   └── scene_graphs.jsonl      # Output file
└── README.md

---

## ⚙️ Setup

Install dependencies:

pip install torch transformers pillow tqdm

(Optional for faster GPU runs)

pip install bitsandbytes accelerate

---

## ▶️ Run

Update paths inside script:

INPUT_FILE = "input/cleaned_vg.jsonl"
IMAGE_DIR = "imagery/"
OUTPUT_FILE = "output/scene_graphs.jsonl"

Run:

python scene_graph_pipeline.py

---

## 📥 Input Format

{
  "image_id": "123",
  "relations": [
    {"subject": "person", "predicate": "holding", "object": "umbrella"}
  ]
}

---

## 📤 Output

- JSONL file with enriched triplets (intent, causal, counterfactual)

---

## 🧠 Notes

- Uses GPU (cuda) by default  
- Checkpoints automatically (safe for long runs)  
- Skips invalid outputs  
