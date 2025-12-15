import json
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Load the edited CSV
csv_path = "ground_truth_descriptions.csv"  # Your edited CSV file from generate_descriptions.py
df = pd.read_csv(csv_path)

# Build LLaVA-style dataset
data = []
for index, row in df.iterrows():
    image_path = row["rotated_path"] if pd.notna(row["rotated_path"]) else row["image_name"]
    description = row["description"]
    angle = row["angle"]
    id_str = f"{Path(image_path).stem}_{angle}"  # Unique ID with angle

    data.append({
        "id": id_str,
        "image": image_path,  # Full path to image
        "conversations": [
            {"from": "human", "value": "<image>\nDescribe this image in detail, including any people, faces, age, ethnicity, gender, and emotional expression."},
            {"from": "gpt", "value": description}
        ]
    })

# Split into train/val/test (80/10/10)
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save JSON files
with open('train.json', 'w') as f: json.dump(train_data, f)
with open('val.json', 'w') as f: json.dump(val_data, f)
with open('test.json', 'w') as f: json.dump(test_data, f)

print("Dataset JSON files created: train.json, val.json, test.json")