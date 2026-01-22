import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ====== 1. 导入 MobileSAM ======
from MobileSAM.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

# ====== 配置路径 ======
# 目标单张图片路径（你指定的路径）
TARGET_IMG_PATH = r"D:\pycharmproject\Uncertainty-aware-Blur-Prior-main\data\things-eeg\Image_set_Resize\train_images\00432_dog\dog_01b.jpg"
TRAIN_SPLIT = "train_images"  # 对应图片路径中的train_images
# 主输出文件夹（所有可视化文件都在这个目录下）
MAIN_OUTPUT_DIR = "weights/sam_train_candidate_masks_visual"
# 子目录：分别存放纯掩码图和网格图
SINGLE_MASK_DIR = os.path.join(MAIN_OUTPUT_DIR, "single_candidate_masks")  # 纯掩码图
GRID_MASK_DIR = os.path.join(MAIN_OUTPUT_DIR, "mask_grids")  # 掩码网格图
# 创建目录（自动创建多级目录）
os.makedirs(SINGLE_MASK_DIR, exist_ok=True)
os.makedirs(GRID_MASK_DIR, exist_ok=True)

# ====== 加载SAM模型 ======
print("🚀 加载 MobileSAM (仅处理单张图片)...")
sam = sam_model_registry["vit_t"](checkpoint="MobileSAM/weights/mobile_sam.pt").eval().cuda()
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.6,
    min_mask_region_area=100,
)


# ====== 核心函数：分类保存纯掩码图+网格图 ======
def visualize_candidate_masks(img_path):
    # 1. 解析文件名（保证唯一性）
    img_basename = os.path.splitext(os.path.basename(img_path))[0]  # dog_01b
    class_name = os.path.basename(os.path.dirname(img_path))  # 00432_dog
    unique_prefix = f"{TRAIN_SPLIT}_{class_name}_{img_basename}"  # train_images_00432_dog_dog_01b

    # 2. 加载图片并生成候选掩码
    pil_img = Image.open(img_path).convert("RGB")
    image_np = np.array(pil_img)
    try:
        candidate_masks = mask_generator.generate(image_np)
        n_masks = len(candidate_masks)
        if n_masks == 0:
            print(f"\n⚠️ {unique_prefix} 无候选掩码，跳过")
            return False
        print(f"📌 {unique_prefix} 生成 {n_masks} 个候选掩码")
    except Exception as e:
        print(f"\n❌ {unique_prefix} 生成掩码失败: {str(e)[:50]}")
        return False

    # 3. 保存纯掩码图（单独子目录）
    for idx, mask_dict in enumerate(candidate_masks):
        # 生成224×224纯掩码图
        mask_ori = mask_dict['segmentation'].astype(np.float32)
        mask_224 = cv2.resize(mask_ori, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask_224_vis = (mask_224 * 255).astype(np.uint8)

        # 纯掩码图命名（含核心属性）
        single_mask_name = f"{unique_prefix}_candidate_{idx}_area{int(mask_dict['area'])}_iou{mask_dict['predicted_iou']:.2f}.png"
        single_mask_path = os.path.join(SINGLE_MASK_DIR, single_mask_name)
        Image.fromarray(mask_224_vis).save(single_mask_path)

        # 可选：保存掩码张量（如需后续模型使用）
        mask_tensor = torch.from_numpy(mask_224)
        tensor_path = os.path.join(SINGLE_MASK_DIR, f"{os.path.splitext(single_mask_name)[0]}.pt")
        torch.save(mask_tensor, tensor_path)

    # 4. 保存掩码网格图（单独子目录）
    n_cols = int(np.ceil(np.sqrt(n_masks)))
    n_rows = int(np.ceil(n_masks / n_cols))
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    gs = GridSpec(n_rows, n_cols, figure=fig)
    fig.suptitle(f"SAM Candidates: {unique_prefix} (Total: {n_masks})", fontsize=12)

    # 绘制网格图（仅纯掩码，无原图）
    for idx, mask_dict in enumerate(candidate_masks):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        mask_224 = cv2.resize(mask_dict['segmentation'].astype(np.float32), (224, 224))
        ax.imshow(mask_224, cmap="gray")
        ax.set_title(f"#{idx}\nArea:{int(mask_dict['area'])}\nIOU:{mask_dict['predicted_iou']:.2f}", fontsize=8)
        ax.axis("off")

    # 填充空白网格
    for idx in range(n_masks, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")

    # 保存网格图
    grid_mask_name = f"{unique_prefix}_all_candidates_grid.png"
    grid_mask_path = os.path.join(GRID_MASK_DIR, grid_mask_name)
    plt.tight_layout()
    plt.savefig(grid_mask_path, dpi=150, bbox_inches="tight")
    plt.close()

    return True


# ====== 主函数：仅处理指定的单张图片 ======
def main():
    # 检查目标图片是否存在
    if not os.path.exists(TARGET_IMG_PATH):
        print(f"❌ 目标图片不存在：{TARGET_IMG_PATH}")
        return

    # 处理单张图片
    print(f"📄 开始处理单张图片：{TARGET_IMG_PATH}")
    success = visualize_candidate_masks(TARGET_IMG_PATH)

    # 输出结果报告
    print("\n" + "=" * 50)
    if success:
        print(f"✅ 单张图片处理完成！")
    else:
        print(f"❌ 单张图片处理失败！")
    print(f"\n📂 输出目录结构：")
    print(f"   主目录：{MAIN_OUTPUT_DIR}")
    print(f"   ├─ 纯掩码图：{SINGLE_MASK_DIR}")
    print(f"   └─ 掩码网格图：{GRID_MASK_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()