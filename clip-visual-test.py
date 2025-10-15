import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageDraw, ImageFont
import os

# ----- CONFIG -----
CLASS_LABELS = ["Crack", "Delamination", "NOdefect"]
IMAGE_FOLDER = r"D:\ShailyDL\DeepL\FINAL_DS_Split\images\test"
SAVE_FOLDER = r"D:\CLIP\visual"
os.makedirs(SAVE_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2

# ----- LOAD MODEL AND PROCESSOR -----
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.load_state_dict(torch.load("clip_finetuned.pth", map_location=DEVICE))
model.eval().to(DEVICE)

# ----- LOAD CLASS TEXT FEATURES -----
text_features = torch.load("clip_text_features.pt", map_location=DEVICE)

# ----- SETUP FONT -----
try:
    font = ImageFont.truetype("arialbd.ttf", 48)  # Arial Bold with size 48
except IOError:
    font = ImageFont.load_default()

# ----- COLLECT IMAGE PATHS -----
image_paths = [
    os.path.join(IMAGE_FOLDER, fname)
    for fname in os.listdir(IMAGE_FOLDER)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
]

# ----- PREDICTION LOOP -----
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

    for path, img, pred, prob in zip(batch_paths, images, preds, probs):
        predicted_class = CLASS_LABELS[pred.item()]
        confidence = prob[pred].item()
        label_text = f"Image classified as: a photo of {predicted_class.lower()} ({confidence:.2f})"

        # ----- CREATE OUTPUT IMAGE WITH WHITE STRIP -----
        img_width, img_height = img.size
        strip_height = 100
        result_img = Image.new("RGB", (img_width, img_height + strip_height), color="white")
        result_img.paste(img, (0, strip_height))

        # ----- DRAW CENTERED, THICK TEXT -----
        draw = ImageDraw.Draw(result_img)
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (img_width - text_width) // 2
        text_y = (strip_height - text_height) // 2

        # Simulate bold by drawing text multiple times
        for offset in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            draw.text((text_x + offset[0], text_y + offset[1]), label_text, fill="black", font=font)

        # ----- SAVE -----
        filename = os.path.basename(path)
        save_path = os.path.join(SAVE_FOLDER, f"classified_{filename}")
        result_img.save(save_path)

        print(f"{filename} ➜ {predicted_class} ({confidence:.2f}) — saved to {save_path}")
