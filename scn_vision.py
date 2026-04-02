#!/usr/bin/env python3
"""
Scene Graph Enrichment Pipeline with Periodic Checkpoint Writes
Writes to JSONL every CHECKPOINT_BATCHES to avoid losing progress.
"""
import json
import re
import os
from tqdm import tqdm
import torch
from PIL import Image

INPUT_FILE = "/data1/sujit/code_dir/NER/Data/datasets/cleaned_vg.jsonl"
IMAGE_DIR = "/data1/sujit/code_dir/NER/Data/datasets/imagery"
OUTPUT_FILE = "scene_graphs_gemma3_1.jsonl"

MAX_SAMPLES = None
MAX_RELATIONS_PER_IMAGE = 5
DEVICE = "cuda:1"
BATCH_SIZE = 64
CHECKPOINT_BATCHES = 10  # Write to JSONL every 10 batches (320 triplets)

MODEL_NAME = "unsloth/gemma-3-12b-it-bnb-4bit"
MAX_NEW_TOKENS_CAUSAL = 12
MAX_NEW_TOKENS_COUNTERFACTUAL = 16
MAX_NEW_TOKENS_INTENT = 8

print("Loading model...")
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.float16
).eval()

print("✓ Model loaded\n")

# ============================
# PROMPTS
# ============================

def causal_prompt(triplet):
    s, p, o = triplet
    return f"""CAUSAL EFFECT: What is the CONSEQUENCE or OUTCOME that results from this relationship?

Relationship: ({s}, {p}, {o})

Think about what HAPPENS or what STATE RESULTS from this. Be specific about the consequence.

Examples with reasoning:
(person, holding, gun) → threat_increase (consequence: danger to others)
(cable, on, floor) → tripping_hazard (consequence: people can trip)
(smoke, from, building) → fire_risk (consequence: building is in danger)
(wet, floor, entrance) → slip_danger (consequence: slippery surface)
(shade, over, sidewalk) → pedestrian_protection (consequence: protection from sun)
(car, parked, street) → obstruction_risk (consequence: blocks path)
(sign, on, building) → visibility_increase (consequence: easier to find)

Generate ONLY the effect/consequence name (1-3 words, underscore-separated, specific noun):"""


def counterfactual_prompt(triplet):
    s, p, o = triplet
    return f"""COUNTERFACTUAL: Modify to create safer/different scenario.

ORIGINAL: ({s}, {p}, {o})

CRITICAL RULES:
1. Subject MUST stay: {s}
2. Only change predicate and object
3. Format: ({s}, NEW_PREDICATE, NEW_OBJECT)

EXAMPLES - NOTE THE SUBJECT NEVER CHANGES:
- (person, holding, gun) → (person, holding, book)
- (cable, on, floor) → (cable, inside, wall)
- (car, moving_fast, road) → (car, parked, road)
- (person, on, ladder) → (person, on, ground)
- (knife, on, table) → (knife, in, drawer)

OUTPUT ONLY: ({s}, predicate, object)
NO ARROWS, NO EXPLANATION, JUST THE MODIFIED TRIPLET."""


def intent_prompt(triplet):
    s, p, o = triplet
    return f"""INTENT: What is the PRIMARY PURPOSE or GOAL of this relationship?

Relationship: ({s}, {p}, {o})

Think about WHY someone would arrange {s} {p} {o}. What is the human intention or goal?

Examples with reasoning:
(person, holding, umbrella) → avoid_rain (goal: stay dry)
(person, typing, keyboard) → work (goal: accomplish tasks)
(person, pointing, gun) → threaten (goal: intimidate)
(person, holding, phone) → communicate (goal: talk to others)
(person, reading, book) → learn (goal: gain knowledge)
(shade, over, sidewalk) → provide_shelter (goal: protect from sun)
(car, parked, street) → temporary_storage (goal: store vehicle)
(sign, on, building) → advertise_business (goal: attract customers)
(light, shining, desk) → illuminate_workspace (goal: enable work)

Generate ONLY the purpose/intent name (1-2 words, underscore-separated, specific action/goal):"""


# ============================
# BATCH GENERATION
# ============================

def generate_batch(prompts, image_paths=None, max_new_tokens=12):
    try:
        outputs = []
        for i, prompt in enumerate(prompts):
            image_path = image_paths[i] if image_paths else None
            
            if image_path and os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]}]
                except:
                    messages = [{"role": "user", "content": [
                        {"type": "text", "text": prompt}
                    ]}]
            else:
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": prompt}
                ]}]
            
            try:
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to(model.device)
                
                input_len = inputs["input_ids"].shape[-1]
                
                with torch.inference_mode():
                    generation = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=False, temperature=1.0, top_p=1.0
                    )
                    generation = generation[0][input_len:]
                
                text = processor.decode(generation, skip_special_tokens=True).strip().lower()
                outputs.append(text if text else "unknown")
                torch.cuda.empty_cache()
            except:
                outputs.append("unknown")
        
        return outputs
    except Exception as e:
        print(f"Batch generation error: {e}")
        return ["unknown"] * len(prompts)


# ============================
# PARSING
# ============================

def clean_text(text):
    text = text.lower().strip()
    for sep in ["```", "=>", "->", "causal:", "effect:", "counterfactual:", "intent:", "purpose:"]:
        text = text.replace(sep, " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(text, max_words=3):
    text = clean_text(text)
    text = re.sub(r"[^a-z0-9_\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "unknown"
    words = text.split()[:max_words]
    label = "_".join(words)
    return re.sub(r"_+", "_", label).strip("_") or "unknown"


def is_valid_label(text):
    return text not in {"", "unknown", "none", "null", "n_a", "na", "vague"} and len(text) > 1


def parse_counterfactual(text, original_triplet):
    text = clean_text(text)
    original_subject = original_triplet[0]
    modified = None

    matches = list(re.finditer(r"\((.*?)\)", text))
    if matches:
        match = matches[-1]
        parts = [x.strip() for x in match.group(1).split(",")]
        if len(parts) >= 3:
            s = parts[0].strip()
            p = parts[1].strip()
            o = ",".join(parts[2:]).strip()
            if s and p and o and len(s) < 50 and len(p) < 50:
                modified = [s, p, o]

    if not modified and "," in text:
        parts = [x.strip() for x in text.split(",")]
        if len(parts) >= 3:
            s = parts[0].strip()
            p = parts[1].strip()
            o = ",".join(parts[2:]).strip()
            if s and p and o and len(s) < 50 and len(p) < 50:
                modified = [s, p, o]

    if not modified:
        modified = original_triplet[:]
    
    if len(modified) >= 1:
        modified[0] = original_subject
    
    is_different = modified != original_triplet
    return modified, is_different


# ============================
# LOAD DATA
# ============================

data = []
with open(INPUT_FILE) as f:
    for i, line in enumerate(f):
        if MAX_SAMPLES is not None and i >= MAX_SAMPLES:
            break
        try:
            data.append(json.loads(line.strip()))
        except:
            pass

print(f"Loaded {len(data)} total images\n")

# Collect all triplets
all_triplets_with_meta = []

for item in data:
    image_id = item.get("image_id", "")
    relations = item.get("relations", [])
    
    image_path_base = os.path.join(IMAGE_DIR, f"{image_id}.jpg") if os.path.exists(IMAGE_DIR) else None
    if image_path_base and not os.path.exists(image_path_base):
        image_path_base = None
    
    for rel in relations[:MAX_RELATIONS_PER_IMAGE]:
        try:
            s = str(rel.get("subject", "")).strip()
            p = str(rel.get("predicate", "")).strip()
            o = str(rel.get("object", "")).strip()

            if not s or not p or not o:
                continue

            triplet = [s, p, o]
            all_triplets_with_meta.append({
                "image_id": image_id,
                "image_path": image_path_base,
                "triplet": triplet
            })
        except:
            continue

print(f"Total triplets to process: {len(all_triplets_with_meta)}\n")

# Clear old output file (fresh start)
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
    print(f"Cleared old {OUTPUT_FILE}\n")

# ============================
# PROCESS IN BATCHES WITH CHECKPOINTS
# ============================

triplet_results = {}
skipped = 0
batch_count = 0

for batch_start in tqdm(range(0, len(all_triplets_with_meta), BATCH_SIZE), desc="Processing"):
    batch_end = min(batch_start + BATCH_SIZE, len(all_triplets_with_meta))
    batch_meta = all_triplets_with_meta[batch_start:batch_end]
    
    triplets = [m["triplet"] for m in batch_meta]
    image_paths = [m["image_path"] for m in batch_meta]
    image_ids = [m["image_id"] for m in batch_meta]
    
    # Generate INTENT batch
    intent_prompts = [intent_prompt(t) for t in triplets]
    intent_texts = generate_batch(intent_prompts, image_paths, MAX_NEW_TOKENS_INTENT)
    intents = [normalize_label(t, max_words=2) for t in intent_texts]
    
    # Generate CAUSAL batch
    causal_prompts = [causal_prompt(t) for t in triplets]
    causal_texts = generate_batch(causal_prompts, image_paths, MAX_NEW_TOKENS_CAUSAL)
    causals = [normalize_label(t, max_words=3) for t in causal_texts]
    
    # Generate COUNTERFACTUAL batch
    cf_prompts = [counterfactual_prompt(t) for t in triplets]
    cf_texts = generate_batch(cf_prompts, image_paths, MAX_NEW_TOKENS_COUNTERFACTUAL)
    counterfactuals = [parse_counterfactual(t, triplets[i]) for i, t in enumerate(cf_texts)]
    
    # Store results
    for i, triplet in enumerate(triplets):
        image_id = image_ids[i]
        intent = intents[i]
        causal = causals[i]
        counterfactual, is_cf_different = counterfactuals[i]
        
        counterfactual[0] = triplet[0]

        if not is_valid_label(intent):
            skipped += 1
            continue

        risk_keywords = {"threat", "risk", "hazard", "danger", "unsafe", "injury", "damage", "collision", "trap"}
        is_risk = any(k in causal.lower() for k in risk_keywords)
        
        if image_id not in triplet_results:
            triplet_results[image_id] = []
        
        triplet_results[image_id].append({
            "triplet": triplet,
            "causal": {
                "effect": causal,
                "type": "risk" if is_risk else "neutral",
                "score": 0.9 if is_risk else 0.5,
                "intent_context": intent
            },
            "counterfactual": {
                "modified_triplet": counterfactual,
                "new_effect": "safer" if is_cf_different else "no_change",
                "change": "decrease" if is_cf_different else "neutral"
            },
            "intent": intent
        })

    batch_count += 1
    
    # CHECKPOINT WRITE every CHECKPOINT_BATCHES
    if batch_count % CHECKPOINT_BATCHES == 0:
        with open(OUTPUT_FILE, "a") as f:
            for image_id in triplet_results:
                item = {"image_id": image_id, "data": triplet_results[image_id]}
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        
        # Clear buffer
        images_written = len(triplet_results)
        triplet_results = {}
        print(f"✓ Checkpoint: Wrote {images_written} images (batch {batch_count})")

# ============================
# FINAL WRITE
# ============================

if triplet_results:
    with open(OUTPUT_FILE, "a") as f:
        for image_id in triplet_results:
            item = {"image_id": image_id, "data": triplet_results[image_id]}
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

print(f"\n{'='*80}")
print(f"Processing complete!")
print(f"Total triplets: {len(all_triplets_with_meta)}")
print(f"Valid: {len(all_triplets_with_meta) - skipped}")
print(f"Skipped: {skipped}")
print(f"Output: {OUTPUT_FILE}")
print(f"{'='*80}\n")

# Check final stats
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        total_lines = sum(1 for _ in f)
    file_size = os.path.getsize(OUTPUT_FILE) / (1024*1024)  # MB
    print(f"Final output: {total_lines} images, {file_size:.2f} MB")