# clip_batch_inference.py
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

print(torch.version.cuda)      # Prints CUDA version PyTorch was built with
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))


# Config
CLASS_LABELS = ["Crack", "Delamination", "NOdefect"]
IMAGE_FOLDER = r"D:\ShailyDL\DeepL\FINAL_DS_Split\images\test"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2  # Adjust based on your GPU memory

# Load processor and model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.load_state_dict(torch.load("clip_finetuned.pth", map_location=DEVICE))

print(f"Using device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
model.eval().to(DEVICE)


# Load class text features
text_features = torch.load("clip_text_features.pt", map_location=DEVICE)

# Gather image paths
image_paths = [
    os.path.join(IMAGE_FOLDER, fname)
    for fname in os.listdir(IMAGE_FOLDER)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
]

# Batch prediction
for i in range(0, len(image_paths), BATCH_SIZE):
    batch_paths = image_paths[i:i + BATCH_SIZE]
    images = [Image.open(p).convert("RGB") for p in batch_paths]

    inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        logits = image_features @ text_features.T
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)

    threshold = 0.5
    # Print results
    for path, pred, prob in zip(batch_paths, preds, probs):
        predicted_class = CLASS_LABELS[pred.item()]
        confidence = prob[pred].item()
        print(f"{os.path.basename(path)} ➜ {predicted_class} (Confidence: {confidence:.2f})")
