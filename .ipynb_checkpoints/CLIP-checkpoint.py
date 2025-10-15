from transformers import CLIPProcessor, CLIPModel
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from PIL import Image
import os
from tqdm import tqdm
import torch

from transformers import CLIPProcessor, CLIPModel

CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").save_pretrained("D:/CLIP/clip_processor")
CLIPModel.from_pretrained("openai/clip-vit-base-patch32").save_pretrained("D:/CLIP/clip_model")

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")


# Initialize processor and model
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

print("Using device:", device)
print("Model is on device:", next(model.parameters()).device)

# Define dataset class
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, class_labels):
        self.root_dir = root_dir
        self.class_labels = class_labels
        self.image_paths = []
        self.labels = []
        
        for label, class_name in enumerate(self.class_labels):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)
                    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        
        # Preprocess image to tensor using CLIP processor
        processed = processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)  # Shape: (3, 224, 224)
        
        return pixel_values, label


# Setup
class_labels = ["Crack", "Delamination", "NOdefect"]
text_prompts = [
    "A surface with a visible crack or not a straight line in middle ",
    "A surface showing delamination damage or peeling",
    "A clean surface with no visible defect"
]
dataset = CustomImageDataset(root_dir=r"D:\ShailyDL\DeepL\CLIPDS", class_labels=class_labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Reduce batch size if running into memory issues

# Optimizer and loss
optimizer = AdamW(model.parameters(), lr=5e-6)
criterion = CrossEntropyLoss()

# Precompute text features for class labels

with torch.no_grad():
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

# Training loop
for epoch in range(20):
    print(f"Starting epoch {epoch+1}", flush=True)
    print(f"\nEpoch {epoch + 1}/{20}")
    model.train()
    total_loss = 0

    for images, labels in tqdm(dataloader, desc=f"Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
    
        image_features = model.get_image_features(pixel_values=images)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        logits = image_features @ text_features.T
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

# Save model and text features after training
torch.save(model.state_dict(), "clip_finetuned.pth")
torch.save(text_features, "clip_text_features.pt")
print("✅ Model and text features saved!")
