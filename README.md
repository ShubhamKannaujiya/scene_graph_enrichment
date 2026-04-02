# Scene Graph Enrichment Pipeline

A Vision-Language Model (VLM) pipeline for enriching Visual Genome scene graphs with causal effects, counterfactual modifications, and intent analysis using Gemma-3-12B.

---

## 📁 Folder Structure

```
project/
├── scn_vision.py              # Main pipeline script
├── README.md                  # This file
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── cleaned_vg.jsonl       # Input scene graphs (Visual Genome)
│   └── imagery/               # Image directory (image_id.jpg files)
│
└── output/
    └── scene_graphs_gemma3_1.jsonl  # Generated enriched scene graphs
```

---

## 🚀 Quick Start

### 1. **Clone/Setup**
```bash
cd /path/to/project
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Prepare Data**
- Place your `cleaned_vg.jsonl` in `data/` directory
- Place images in `data/imagery/` (named as `{image_id}.jpg`)

### 4. **Run Pipeline**
```bash
python scn_vision.py
```





## 📤 Output Format

**Output JSONL** (`scene_graphs_gemma3_1.jsonl`):
```json
{
  "image_id": "123456",
  "data": [
    {
      "triplet": ["person", "holding", "gun"],
      "causal": {
        "effect": "threat_increase",
        "type": "risk",
        "score": 0.9,
        "intent_context": "intimidate"
      },
      "counterfactual": {
        "modified_triplet": ["person", "holding", "book"],
        "new_effect": "safer",
        "change": "decrease"
      },
      "intent": "intimidate"
    }
  ]
}
```



## 📈 Output Statistics

After completion, the script prints:
```
================================================================================
Processing complete!
Total triplets: 12345
Valid: 10567
Skipped: 1778
Output: scene_graphs_gemma3_1.jsonl
================================================================================

Final output: 5432 images, 250.45 MB
```

---
