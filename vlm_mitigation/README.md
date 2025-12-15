VLM Mitigation

This folder contains all scripts, datasets, and utilities used for bias mitigation in Vision-Language Models (VLMs), specifically for models such as LLaVA, LLaVA-1.5, and related LVLM architectures.

📌 Overview

The goal of this module is to reduce demographic and representational bias in VLM outputs through:

🔹 Fine-tuning (LoRA or full fine-tuning)

🔹 Prompt-level mitigation

🔹 Data balancing

🔹 Controlled image–text training

🔹 Evaluation before/after mitigation

This work is part of the Vision-Language Model Bias Mitigation section of my Master’s Thesis.

📁 Folder Contents
1. Fine-Tuning Scripts

python_finetune_llava.py
Script for fine-tuning LLaVA using LoRA or full-parameter training.

2. Inference & Testing

python_llava_infer.py
Run inference on images before/after mitigation.

python_test_3_images.py
Evaluate model robustness using randomly rotated images or multiple viewpoints.

3. Dataset Files

train.json

val.json

test.json

These JSON files follow the standard LLaVA/BLIP captioning format and contain:

Image paths

Demographic labels

Balanced / augmented captions

Bias-corrected text prompts

🧪 Evaluation Metrics

Mitigation effectiveness is evaluated using:

✔ Sensitivity to demographic attributes

✔ Model probability / response shift

✔ Toxicity & bias scoring

✔ Consistency under rotation or data augmentation

Results compare:

Baseline VLM

Mitigated (fine-tuned) VLM

Mitigated (prompt-engineered) VLM

🚀 Usage Instructions
Fine-tuning
python python_finetune_llava.py --config train.json

Inference
python python_llava_infer.py --image path/to/image.jpg

Testing Robustness
python python_test_3_images.py

📄 Notes

Image datasets are not included in this repository due to size and copyright restrictions.

JSON files reference local image paths that must match your dataset directory.

📚 Citation / Thesis Link

This code supports my Master’s Thesis on Bias Detection and Mitigation in Vision-Language Models (VLMs).This folder will contain scripts for bias mitigation in Vision-Language Models.
