import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import easyocr
import numpy as np
import re
import os
import io


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
MODEL_PATH = os.path.join(BASE_DIR, "model", "multimodal_fold_5.pth")

# =========================
# 1. Text Preprocessing
# =========================
def preprocess_text(text):
    emoji_pattern = re.compile(
        "["  
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U00002600-\U000026FF"  # misc symbols
        "\U00002B00-\U00002BFF"  # arrows, etc.
        "\U0001FA70-\U0001FAFF"  # extended symbols
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b\w\b', '', text)
    return text

# =========================
# 2. OCR Extraction
# =========================
def ocr_extract_text(image_path):
    reader = easyocr.Reader(['en', 'tl'], gpu=torch.cuda.is_available())
    result = reader.readtext(image_path, detail=0)
    final_text = " ".join(result)
    preprocess_txt = preprocess_text(final_text)
    return final_text, preprocess_txt

# =========================
# 3. Image Preprocessing
# =========================
def resize_normalize_image(image_path, target_size=(224, 224)):
    preprocess_image = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # image = Image.open(image_path).convert('RGB')
    # image = preprocess_image(image)
    # return image.unsqueeze(0)  # add batch dim
    if isinstance(image_path, str):  # file path
        pil_image = Image.open(image_path).convert('RGB')
    elif isinstance(image_path, Image.Image):  # already a PIL image
        pil_image = image_path.convert('RGB')
    elif isinstance(image_path, (bytes, bytearray)):  # raw bytes
        pil_image = Image.open(io.BytesIO(image_path)).convert('RGB')
    else:
        raise TypeError(f"Unsupported type {type(image_path)} for resize_normalize_image")

    return preprocess_image(pil_image).unsqueeze(0)  # add batch dim
# =========================
# 4. Model Definitions
# =========================
class CrossAttentionModule(nn.Module):
    def __init__(self, image_dim=512, text_dim=768, hidden_dim=256):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, image_features, text_features):
        image_proj = self.image_proj(image_features).unsqueeze(0)
        text_proj = self.text_proj(text_features).unsqueeze(0)
        attended_image, _ = self.attention(image_proj, text_proj, text_proj)
        attended_image = self.layer_norm(attended_image + image_proj)
        attended_text, _ = self.attention(text_proj, image_proj, image_proj)
        attended_text = self.layer_norm(attended_text + text_proj)
        return attended_image.squeeze(0), attended_text.squeeze(0)

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes=2, model_name='jcblaise/roberta-tagalog-base'):
        super().__init__()
        # Image encoder
        self.image_encoder = models.resnet18(pretrained=True)
        self.image_encoder.fc = nn.Identity()

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(model_name)

        # Cross-attention
        self.cross_attention = CrossAttentionModule(
            image_dim=512,
            text_dim=self.text_encoder.config.hidden_size,
            hidden_dim=256
        )

        # Fusion + classification
        self.fusion = nn.Sequential(
            nn.Linear(256*2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        img_features = self.image_encoder(images)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_out.pooler_output
        attended_img, attended_text = self.cross_attention(img_features, text_features)
        fused_features = torch.cat([attended_img, attended_text], dim=1)
        return self.fusion(fused_features)

# =========================
# 5. Load Model & Tokenizer
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MultimodalClassifier(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("jcblaise/roberta-tagalog-base")

# =========================
# 6. Inference Function
# =========================
def run_inference(image_path):
    # Convert bytes → PIL image
    pil_img = Image.open(io.BytesIO(image_path)).convert("RGB")

    # OCR
    np_image= np.array(pil_img)
    raw_text, clean_text = ocr_extract_text(np_image)

    # Image
    img_tensor = resize_normalize_image(pil_img).to(device)

    # Tokenize text
    encoding = tokenizer(
        clean_text, return_tensors='pt',
        padding=True, truncation=True, max_length=128
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(img_tensor, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_class = 'sexual' if pred_class == 1 else 'non-sexual'

    return {
        'prediction': pred_class,
        'probabilities': probs.cpu().numpy().tolist(),
        'raw_text': raw_text,
        'clean_text': clean_text
    }

# =========================
# 7. Visualization
# =========================
# def visualize_result(image_path, result):
#     img = Image.open(image_path)
#     plt.figure(figsize=(6,6))
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title(f"Prediction: {result['prediction']} | Prob: {result['probabilities'][0]}")
#     plt.show()

# =========================
# 8. Example Usage
# =========================
# if __name__ == "__main__":
#     image_path = "backend/sexual_text.jpg"
#     result = run_inference(image_path)
#     print("\n--- Inference Result ---")
#     print("Predicted Class:", result['prediction'])
#     print("Probabilities:", result['probabilities'])
#     print("OCR Raw Text:", result['raw_text'])
#     print("Cleaned Text:", result['clean_text'])

#     visualize_result(image_path, result)