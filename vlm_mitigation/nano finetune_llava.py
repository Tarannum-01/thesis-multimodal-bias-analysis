import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'  # For XLA/CUDA
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from trl import SFTTrainer
from PIL import Image

# Model and processor loading
model_name = "llava-hf/llava-1.5-7b-hf"
processor = LlavaProcessor.from_pretrained(model_name, trust_remote_code=True)
model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.to("cuda")

# Dataset paths
data_root = "./Tuning"
train_json = os.path.join(data_root, "train.json")
val_json = os.path.join(data_root, "val.json")

# Load dataset
dataset = load_dataset(
    "json",
    data_files={"train": train_json, "validation": val_json},
)

# Preprocess function
def preprocess(examples):
    images = []
    for img_path in examples["image"]:
        if img_path.startswith("C:\\Users\\taran"):
            filename = os.path.basename(img_path)
            img_path = f"./fine tune/{filename}"
        try:
            images.append(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            images.append(Image.new("RGB", (224, 224), color=(255, 255, 255)))
 
    prompts = [conv[0]["value"].replace("<image>", "") for conv in examples["conversations"]]
    targets = [conv[1]["value"] for conv in examples["conversations"]]
 
    inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt")
    labels = processor(text=targets, padding=True, return_tensors="pt").input_ids
 
    inputs["labels"] = labels
    return inputs

# Map preprocessing
train_dataset = dataset["train"].map(preprocess, batched=True, num_proc=1)
val_dataset = dataset["validation"].map(preprocess, batched=True, num_proc=1)

# Training args
training_args = TrainingArguments(
    output_dir="./finetuned_vlm_gpu",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    remove_unused_columns=False,
    use_cpu=False,
    dataloader_num_workers=2,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer,
)

# Train and save
trainer.train()
trainer.save_model("./finetuned_vlm_gpu")
processor.save_pretrained("./finetuned_vlm_gpu")
print("Fine-tuning complete. Model saved in ./finetuned_vlm_gpu.")