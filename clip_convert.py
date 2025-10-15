from transformers import CLIPModel, CLIPConfig
from transformers import CLIPProcessor, CLIPModel
import torch

# Load model config (or modify if custom)
config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")

# Create new model and load weights manually
model = CLIPModel(config)
model.load_state_dict(torch.load("D:/CLIP/clip_finetuned.pth"))

# Save in Hugging Face format
model.save_pretrained("D:/CLIP/clip_finetuned_hf")
