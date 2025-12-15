HuggingFace-Based Image Generation

This folder contains scripts for generating synthetic demographic images using HuggingFace diffusion models. These images were used for testing bias behavior in Vision-Language Models and for comparing robustness across generative pipelines.

Files

demographic_HF_image.py

generate_hf_image.py

⚠️ Note: Generated images are not included in this repository to keep the repo lightweight.

Requirements

Install the following Python packages:

diffusers
transformers
accelerate
torch

Usage

Install the required libraries

Run the scripts to generate demographic test images

Output images will be saved locally to your chosen directory
