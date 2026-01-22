# gen_mask_sam_clip_meg.py

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
import open_clip
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SAM+CLIP masks for MEG images")
    parser.add_argument("--image_root", type=str, default="data/things-meg/Image_set_Resize",
                        help="Path to resized MEG image folder (e.g., Image_set_Resize)")
    parser.add_argument("--output_dir", type=str, default="weights/meg/masks_sam_clip",
                        help="Directory to save .pt mask files")
    parser.add_argument("--sam_checkpoint", type=str, default="MobileSAM/weights/mobile_sam.pt",
                        help="Path to SAM model checkpoint")
    parser.add_argument("--sam_model_type", type=str, default="vit_t", choices=["vit_t", "vit_l", "vit_b"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()

    # --- 1. 创建输出目录 ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. 加载 SAM ---
    print("Loading SAM...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # 去除小噪声区域
    )

    # --- 3. 加载 CLIP ---
    print("Loading CLIP...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    clip_model = clip_model.to(args.device).eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    clip_normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )

    # --- 4. 获取所有图像路径 ---
    all_images = []
    for root, _, files in os.walk(args.image_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                rel_path = os.path.relpath(os.path.join(root, f), args.image_root)
                all_images.append(rel_path)

    print(f"Found {len(all_images)} images.")

    # --- 5. 处理每张图 ---
    for img_rel_path in tqdm(all_images, desc="Generating masks"):
        img_path = os.path.join(args.image_root, img_rel_path)
        output_path = os.path.join(args.output_dir, os.path.splitext(img_rel_path)[0] + ".pt")

        # 跳过已存在的
        if os.path.exists(output_path):
            continue

        try:
            # 加载原图
            image_pil = Image.open(img_path).convert("RGB")
            image_np = np.array(image_pil)

            # SAM 生成多个 masks
            masks = mask_generator.generate(image_np)
            if not masks:
                # 若无 mask，用全1
                final_mask = torch.ones((224, 224), dtype=torch.float32)
            else:
                # CLIP 图像编码（原始图）
                with torch.no_grad():
                    orig_tensor = clip_normalize(transforms.ToTensor()(image_pil.resize((224, 224)))).unsqueeze(0).to(
                        args.device)
                    orig_feat = clip_model.encode_image(orig_tensor)
                    orig_feat /= orig_feat.norm(dim=-1, keepdim=True)

                best_score = -1
                best_mask = None

                for mask_dict in masks:
                    mask = mask_dict['segmentation']  # H x W bool
                    # 将 mask 应用于原图：前景保留，背景变灰或黑
                    masked_img = image_np.copy()
                    masked_img[~mask] = 0  # 黑色背景（也可用均值）
                    masked_pil = Image.fromarray(masked_img).resize((224, 224))

                    # CLIP 编码 masked 图
                    with torch.no_grad():
                        masked_tensor = clip_normalize(transforms.ToTensor()(masked_pil)).unsqueeze(0).to(args.device)
                        masked_feat = clip_model.encode_image(masked_tensor)
                        masked_feat /= masked_feat.norm(dim=-1, keepdim=True)

                    # 计算相似度
                    sim = (orig_feat @ masked_feat.T).item()
                    if sim > best_score:
                        best_score = sim
                        best_mask = mask.astype(np.float32)

                # 调整到 224x224（与模型输入一致）
                best_mask_pil = Image.fromarray(best_mask).resize((224, 224), Image.NEAREST)
                final_mask = torch.from_numpy(np.array(best_mask_pil)).float()

            # 保存
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(final_mask, output_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # 出错时保存全1掩码
            final_mask = torch.ones((224, 224), dtype=torch.float32)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(final_mask, output_path)

    print("✅ All masks generated.")


if __name__ == "__main__":
    main()