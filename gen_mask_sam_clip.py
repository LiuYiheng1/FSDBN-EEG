import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Resize
import torch.nn.functional as F
from tqdm import tqdm

# ====== 1. 导入 MobileSAM ======
from MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

# ====== 2. 导入 CLIP ======
from clipmain.clipset import clip

# ====== 配置路径 ======
IMG_ROOT = r"D:\pycharmproject\Uncertainty-aware-Blur-Prior-main\data\things-eeg\Image_set"
MASK_DIR = "weights/masks_sam_clip"  # 新目录，避免覆盖旧掩码
os.makedirs(MASK_DIR, exist_ok=True)

# ====== 加载模型 ======
print("🚀 加载 MobileSAM...")
sam = sam_model_registry["vit_t"](checkpoint="MobileSAM/weights/mobile_sam.pt").eval().cuda()
mask_generator = SamAutomaticMaskGenerator(sam)

print("🖼️ 加载 CLIP (ViT-B/32)...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

resize_224 = Resize((224, 224), antialias=True)

def mask_to_bbox(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return (int(x_min), int(y_min), int(x_max), int(y_max))

def generate_foreground_mask(img_path):
    pil_img = Image.open(img_path).convert("RGB")
    image_np = np.array(pil_img)

    # Step 1: 获取整图 CLIP 特征
    try:
        full_input = clip_preprocess(pil_img).unsqueeze(0).cuda()
        with torch.no_grad():
            full_feat = clip_model.encode_image(full_input).float()  # [1, 512]
    except Exception as e:
        print(f"⚠️ CLIP 整图特征失败: {e}")
        full_feat = None

    # Step 2: SAM 分割
    masks = []
    try:
        masks = mask_generator.generate(image_np)
    except Exception as e:
        print(f"⚠️ SAM 分割失败: {e}")
        masks = []

    best_mask = None
    best_score = -1.0

    # Step 3: 如果有整图特征，用 CLIP 打分选最佳 mask
    if full_feat is not None and masks:
        for m in masks:
            bbox = mask_to_bbox(m['segmentation'])
            if bbox is None:
                continue
            try:
                cropped = pil_img.crop(bbox)
                crop_input = clip_preprocess(cropped).unsqueeze(0).cuda()
                with torch.no_grad():
                    crop_feat = clip_model.encode_image(crop_input).float()
                sim = F.cosine_similarity(full_feat, crop_feat).item()
                if sim > best_score:
                    best_score = sim
                    best_mask = m['segmentation']
            except Exception:
                continue  # 跳过无效 crop

    # Step 4: 如果没选出 best_mask，用最大面积 fallback
    if best_mask is None and masks:
        largest = max(masks, key=lambda x: x['area'])
        best_mask = largest['segmentation']

    # Step 5: 如果 still None，用 center crop
    if best_mask is None:
        w, h = pil_img.size
        mask = np.zeros((h, w), dtype=np.float32)
        crop_w, crop_h = int(w * 0.6), int(h * 0.6)
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        mask[y1:y1+crop_h, x1:x1+crop_w] = 1.0
        best_mask = mask

    # Step 6: 转为 tensor 并 resize 到 224x224
    mask_tensor = torch.from_numpy(best_mask).float()
    if mask_tensor.ndim == 2:
        mask_tensor = resize_224(mask_tensor.unsqueeze(0)).squeeze(0)
    else:
        mask_tensor = resize_224(mask_tensor)

    return mask_tensor

# ====== 主循环 ======
total_masks = 0

for split in ["training_images", "test_images"]:
    split_path = os.path.join(IMG_ROOT, split)
    if not os.path.exists(split_path):
        print(f"⚠️ 路径不存在: {split_path}")
        continue

    class_dirs = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    print(f"📁 {split}: 共 {len(class_dirs)} 个类别")

    for class_name in tqdm(class_dirs, desc=f"处理 {split}"):
        class_path = os.path.join(split_path, class_name)
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith('.jpg'):
                continue

            img_path = os.path.join(class_path, img_file)
            mask_name = os.path.splitext(img_file)[0] + ".pt"
            mask_save_path = os.path.join(MASK_DIR, mask_name)

            try:
                mask_f = generate_foreground_mask(img_path)
                torch.save(mask_f, mask_save_path)
                total_masks += 1
            except Exception as e:
                print(f"❌ 完全失败: {img_path} | {e}")

print(f"✅ 全部完成！共生成 {total_masks} 个高质量前景掩码。")


