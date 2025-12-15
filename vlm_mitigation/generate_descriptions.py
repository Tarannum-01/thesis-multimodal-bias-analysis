import pandas as pd
from pathlib import Path
from PIL import Image
import base64
import requests
import os

class OllamaImageDescriptor:
    def __init__(self, base_url: str = "http://localhost:11435"):
        self.base_url = base_url.rstrip('/')

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            encoded_str = base64.b64encode(img_file.read()).decode("utf-8")
        return encoded_str

    def describe_image(self, image_path: str, model: str = "gemma3:4b") -> str:
        image_b64 = self.encode_image(image_path)
        payload = {
            "model": model,
            "prompt": "Describe this image in detail, including any people, faces, age, ethnicity, gender, and emotional expression.",
            "images": [image_b64],
            "stream": False
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            if response.status_code == 200:
                return response.json().get("response", "No description found.")
            elif response.status_code == 500:
                return "500 Server Error: Try reducing image size, spacing out requests, or restarting Ollama."
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Request failed: {str(e)}"

def generate_angle_descriptions(image_folder, output_csv, angles=[0, 90, 180], num_images=None):
    descriptor = OllamaImageDescriptor()
    folder = Path(image_folder)
   
    # Get all images in folder
    image_files = list(folder.glob("*.jpeg")) + list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
   
    if num_images is not None:
        image_files = image_files[:num_images]
   
    data = []
   
    for image_path in image_files:
        print(f"Processing image: {image_path.name}")
        for angle in angles:
            try:
                img = Image.open(image_path)
                if angle != 0:
                    img_rotated = img.rotate(angle, expand=True)  # Positive angle for counterclockwise rotation
                    rotated_path = str(folder / f"rotated_{angle}_{image_path.name}")
                    img_rotated.save(rotated_path)
                    description = descriptor.describe_image(rotated_path)
                    data.append({
                        "image_name": image_path.name,
                        "angle": angle,
                        "description": description,
                        "rotated_path": rotated_path
                    })
                    print(f"Description ({angle}° rotation):\n{description}")
                else:
                    description = descriptor.describe_image(str(image_path))
                    data.append({
                        "image_name": image_path.name,
                        "angle": angle,
                        "description": description,
                        "rotated_path": str(image_path)
                    })
                    print(f"Original Image Description:\n{description}")
                print("-" * 80)
            except Exception as e:
                print(f"Could not process {image_path.name} at {angle}°: {e}")
   
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Descriptions saved to {output_csv}")

if __name__ == "__main__":
    image_folder = r"C:\Users\taran\OneDrive\Documents\thesis doc\Augmentation\fine tune"  # Updated based on your latest path
    output_csv = "descriptions.csv"  # Will save in the script's directory
    generate_angle_descriptions(image_folder, output_csv)