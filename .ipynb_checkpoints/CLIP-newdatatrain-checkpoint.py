from transformers import CLIPProcessor, CLIPModel
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from PIL import Image
import os
from tqdm import tqdm

# ==========================
# Paths (SAFE - new names only)
# ==========================
MODEL_NAME = "openai/clip-vit-base-patch32"
OLD_FINETUNED_PATH = "clip_finetuned.pth"   # your old checkpoint (read only)
NEW_DATASET_IMAGES = r"D:\Deep_Learning\Dataset\Phone_crack_data_partial\images\train"
NEW_DATASET_LABELS = r"D:\Deep_Learning\Dataset\Phone_crack_data_partial\labels\train"

# new save names (will not overwrite old files)
NEW_MODEL_PATH = "clip_finetuned_resumed_v4.pth"
NEW_EMBEDDINGS_PATH = "clip_trained_embeddings_v4.pt"

# ==========================
# Device setup
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================
# Load processor & model
# ==========================
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)

# Load old fine-tuned weights (only if exist)
if os.path.exists(OLD_FINETUNED_PATH):
    state_dict = torch.load(OLD_FINETUNED_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded old fine-tuned weights from {OLD_FINETUNED_PATH}")
else:
    print("⚠️ No old fine-tuned weights found, starting from base CLIP.")

print("Model is on device:", next(model.parameters()).device)

# ==========================
# Dataset Class (YOLO label format)
# ==========================
class YoloImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, class_labels):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.class_labels = class_labels
        self.image_paths = []
        self.labels = []

        for img_name in os.listdir(self.image_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.image_dir, img_name)
                label_path = os.path.join(
                    self.label_dir,
                    os.path.splitext(img_name)[0] + ".txt"
                )

                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            # take the first class_id (image-level label)
                            class_id = int(lines[0].split()[0])
                            self.image_paths.append(img_path)
                            self.labels.append(class_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        processed = processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)
        return pixel_values, label

# ==========================
# Setup dataset
# ==========================
class_labels = ["Crack", "Delamination", "NOdefect"]
text_prompts = [
    "A surface with a visible crack or not a straight line in middle",
    "A surface showing delamination damage or peeling",
    "A clean surface with no visible defect"
]

dataset = YoloImageDataset(NEW_DATASET_IMAGES, NEW_DATASET_LABELS, class_labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ==========================
# Optimizer & loss
# ==========================
optimizer = AdamW(model.parameters(), lr=5e-6)
criterion = CrossEntropyLoss()

# ==========================
# Precompute text embeddings (before training for logits calc)
# ==========================
with torch.no_grad():
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

# ==========================
# Training loop
# ==========================
EPOCHS = 50

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
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

# ==========================
# Save updated model + embeddings (new files only)
# ==========================
torch.save(model.state_dict(), NEW_MODEL_PATH)

model.eval()
with torch.no_grad():
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
    updated_text_features = model.get_text_features(**text_inputs)
    updated_text_features = updated_text_features / updated_text_features.norm(p=2, dim=-1, keepdim=True)

torch.save(updated_text_features, NEW_EMBEDDINGS_PATH)

print(f"✅ New model saved at {NEW_MODEL_PATH}")
print(f"✅ Updated embeddings saved at {NEW_EMBEDDINGS_PATH}")
