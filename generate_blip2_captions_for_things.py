# generate_blip2_captions_for_things.py
import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration





# ================== 配置 ==================
IMAGE_ROOT = r"\data\things-meg\Image_set"
OUTPUT_JSON = r"\weights\event_captions_blip2.json"

MAX_NEW_TOKENS = 20
BATCH_SIZE = 1  # BLIP-2 不支持多图 batch，设为 1
# =========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
LOCAL_MODEL_PATH = r"\models\blip2-opt-2.7b"

processor = Blip2Processor.from_pretrained(LOCAL_MODEL_PATH)
model = Blip2ForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)
# 加载 BLIP-2
print("Loading BLIP-2 model...")

model.eval()

# 收集所有图像路径
all_image_paths = []
for split in ["training_images", "test_images"]:
    split_dir = os.path.join(IMAGE_ROOT, split)
    if not os.path.exists(split_dir):
        continue
    for class_folder in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.join(split, class_folder, img_file).replace("\\", "/")
                all_image_paths.append(rel_path)

print(f"Found {len(all_image_paths)} images.")

# 生成 captions
captions = {}
for rel_path in tqdm(all_image_paths, desc="Generating BLIP-2 captions"):
    full_path = os.path.join(IMAGE_ROOT, rel_path)
    try:
        image = Image.open(full_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        captions[rel_path] = caption
    except Exception as e:
        print(f"Error processing {full_path}: {e}")
        captions[rel_path] = ""  # 或跳过

# 保存
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
# 1. 保存 json（原功能保留）
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(captions, f, indent=2, ensure_ascii=False)
print(f"✅ Saved {len(captions)} captions to {OUTPUT_JSON}")

# 2. 新增：caption → Sentence-BERT 384 维向量
print("Encoding captions with Sentence-BERT ...")
from sentence_transformers import SentenceTransformer
import torch

sbert = SentenceTransformer('all-MiniLM-L6-v2')   # 384 维，最快
sbert_vecs = {}
for path, cap in tqdm(captions.items(), desc="SBERT encoding"):
    vec = sbert.encode(cap, normalize=True)       # ndarray (384,)
    sbert_vecs[path] = torch.from_numpy(vec).half()

sb_pt = OUTPUT_JSON.replace("event_captions_blip2.json", "caption_sbert384.pt")
torch.save(sbert_vecs, sb_pt)
print(f"✅ Saved SBERT vectors to {sb_pt}")



