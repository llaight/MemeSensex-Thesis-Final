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
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_multimodal.pth")

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
    def __init__(self, image_dim=512, text_dim=768, hidden_dim=256, num_heads=8):
        super(CrossAttentionModule, self).__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.attn_img2txt = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.attn_txt2img = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, image_feats, text_feats):
        # Project
        img_proj = self.image_proj(image_feats)   # (B, num_patches, hidden)
        txt_proj = self.text_proj(text_feats)     # (B, seq_len, hidden)

        # Image attends to text
        img_attn, _ = self.attn_img2txt(img_proj, txt_proj, txt_proj)
        img_out = self.norm1(img_proj + img_attn)

        # Text attends to image
        txt_attn, _ = self.attn_txt2img(txt_proj, img_proj, img_proj)
        txt_out = self.norm2(txt_proj + txt_attn)

        return img_out, txt_out

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes=2, model_name='jcblaise/roberta-tagalog-base',
                 use_mean_pooling=False):
        super(MultimodalClassifier, self).__init__()

        # Image encoder (ResNet-18, keep spatial features)
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]  # keep until last conv (before avgpool)
        self.image_encoder = nn.Sequential(*modules)  # (B, 512, 7, 7)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.use_mean_pooling = use_mean_pooling

        # Cross-Attention
        self.cross_attention = CrossAttentionModule(
            image_dim=512,
            text_dim=self.text_encoder.config.hidden_size,
            hidden_dim=256,
            num_heads=8
        )

        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # Extract image features
        B = images.size(0)
        img_feats = self.image_encoder(images)           # (B, 512, 7, 7)
        img_feats = img_feats.flatten(2).permute(0, 2, 1)  # (B, 49, 512)

        # Extract text features
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_mean_pooling:
            mask = attention_mask.unsqueeze(-1).expand(text_outputs.last_hidden_state.size())
            sum_hidden = torch.sum(text_outputs.last_hidden_state * mask, dim=1)
            sum_mask = mask.sum(1).clamp(min=1e-9)
            txt_feats = sum_hidden / sum_mask
            txt_feats = txt_feats.unsqueeze(1)
        else:
            txt_feats = text_outputs.last_hidden_state  # (B, seq_len, H)

        attended_img, attended_txt = self.cross_attention(img_feats, txt_feats)

        img_repr = attended_img.mean(dim=1)  # (B, hidden)
        txt_repr = attended_txt[:, 0, :]     # CLS

        fused = torch.cat([img_repr, txt_repr], dim=1)
        return self.fusion(fused)

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

