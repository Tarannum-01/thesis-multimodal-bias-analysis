VLM Bias & Robustness Testing

This folder contains scripts for evaluating bias, consistency, and robustness in Vision–Language Models (VLMs) under different image transformations and conditions.

Included Tests
1. Angle Rotation Bias Test

image_angle_test.py

Tests how VLM descriptions change when an image is rotated (0°, 45°, 90°, 180°).

Detects instability, hallucinations, and demographic misclassification caused by rotation.

 2. General Image Consistency Test

image_test.py

Feeds the same image to the VLM multiple times.

Measures response variance and bias drift.

 3. Random Demographic Bias Test

random_imagetest.py

Uses randomly selected human face images.

Evaluates whether the VLM produces:

Incorrect ethnicity descriptions

Stereotypical or biased labels

Inconsistent outputs

Input Folders

The following zipped folders contain the images used in each test:

image_angle_test/

Image_test/

random_Image_test/

Purpose

These scripts support the thesis “Bias Detection and Robustness Evaluation in Multimodal AI Systems.”
They are specifically designed to expose:

Demographic bias

Angle-induced misclassification

VLM hallucination behavior

Sensitivity to image perturbations

how to run
python image_angle_test.py
python image_test.py
python random_imagetest.py


These scripts are designed to test ANY Vision-Language Model.
In your thesis work, you used:

 Gemma-3 Vision (via Ollama or HF Transformers)

Gemma3:4b

Gemma3:12b

Local inference using Ollama or transformers pipeline

Supported Models

You may test:

✔ LLaVA
✔ LLaVA-1.5
✔ Gemini-Vision (if API available)
✔ Gemma-3 Vision
✔ Any VLM supporting image input

 Expected Output

Each script produces:

JSON or text logs of predictions

Side-by-side comparisons

Highlighted bias or instability



