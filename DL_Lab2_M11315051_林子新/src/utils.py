import numpy as np
import matplotlib.pyplot as plt
import cv2


def dice_score(pred_mask, gt_mask, threshold=0.5):
    """
    计算二值化预测和真实标注的 Dice Score
    
    参数：
    - pred_mask: 预测的分割掩码，可以是概率图或二值图（numpy数组）
    - gt_mask:   真实的分割标注（numpy数组）
    - threshold: 如果pred_mask为概率图，则用此阈值转换为二值图（默认0.5）
    
    返回：
    - dice: Dice Score值，介于0和1之间
    """
    # 如果预测为概率图，则进行二值化处理
    if pred_mask.dtype != np.bool and pred_mask.dtype != np.int8:
        pred_mask = (pred_mask > threshold).astype(np.float32)
    else:
        pred_mask = pred_mask.astype(np.float32)
        
    # 同理，确保gt_mask为浮点类型
    if gt_mask.dtype != np.bool and gt_mask.dtype != np.int8:
        gt_mask = (gt_mask > threshold).astype(np.float32)
    else:
        gt_mask = gt_mask.astype(np.float32)
    
    # 展平数组以便计算
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # 计算交集和总和
    intersection = np.sum(pred_flat * gt_flat)
    total = np.sum(pred_flat) + np.sum(gt_flat)
    
    # 如果预测和真实均为空，则定义Dice Score为1（完美匹配）
    if total == 0:
        return 1.0
    
    dice = 2.0 * intersection / total
    return dice

def plot_training_metrics(epochs, train_losses, val_losses, train_dices, val_dices):
    """
    繪製兩個子圖：
    - 左圖：Loss 隨 Epoch 變化（包含訓練與驗證 Loss）
    - 右圖：Dice Score 隨 Epoch 變化（包含訓練與驗證 Dice Score）
    並保存為 PNG 檔案
    """
    plt.figure(figsize=(12, 5))
    
    # 左圖：Loss 隨 Epoch 變化
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    
    # 右圖：Dice Score 隨 Epoch 變化
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_dices, label='Train Dice', marker='o', color='blue')
    plt.plot(epochs, val_dices, label='Validation Dice', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Dice Score vs Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=300)
    plt.show()

def visualize_mask_black(image, mask, alpha=1, bgr2rgb=False):
    """
    將輸入影像中的 mask 區域以黑色填充。
    
    參數：
    - image: 原始影像 (numpy陣列, H x W x 3)，可能是 BGR 或 RGB
             值域可為 [0,1] 或 [0,255]
    - mask:  二值遮罩 (numpy陣列, H x W)，mask>0 表示有物體
    - alpha: 混合比例
             alpha=1 時完全不透明 (覆蓋)
             alpha<1 時使用加權混合
    - bgr2rgb: 若為 True，將會把輸入影像視為 BGR，轉為 RGB 後再處理，
               避免 Matplotlib 中顏色顛倒。

    回傳：
    - result: (numpy陣列, H x W x 3, uint8)，已將 mask 區域填成黑色
    """
    # 若需要，先將 BGR 轉為 RGB
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 確保影像為 uint8
    if image.dtype != np.uint8:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.copy()
    
    # 建立 overlay，將遮罩區域填黑
    overlay = image_uint8.copy()
    mask_binary = (mask > 0)
    overlay[mask_binary] = [0, 0, 0]
    
    # 若 alpha=1，直接返回覆蓋後的結果
    if alpha >= 1:
        return overlay
    else:
        # 否則使用加權混合
        result = cv2.addWeighted(image_uint8, 1 - alpha, overlay, alpha, 0)
        return result

def visualize_comparison(image, gt_mask, pred_mask, alpha=1, bgr2rgb=False):
    """
    在同一張圖中顯示：
    1) 原始輸入影像
    2) 疊加了 Ground Truth Mask 的影像（黑色填充）
    3) 疊加了 Predicted Mask 的影像（黑色填充）

    並將結果存成 PNG 檔
    """
    # 如果 GT 或 Pred Mask 為 None，則用全零遮罩代替
    if gt_mask is None:
        gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if pred_mask is None:
        pred_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 產生 GT 與預測疊圖
    gt_overlay = visualize_mask_black(image, gt_mask, alpha=alpha, bgr2rgb=bgr2rgb)
    pred_overlay = visualize_mask_black(image, pred_mask, alpha=alpha, bgr2rgb=bgr2rgb)

    # 如果要在同一個函式中顯示原圖，也要做 bgr2rgb（否則 Matplotlib 會顏色顛倒）
    if bgr2rgb:
        # 先把原圖也轉成 RGB
        if image.dtype != np.uint8:
            tmp = (image * 255).astype(np.uint8)
        else:
            tmp = image.copy()
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        image_to_show = tmp
    else:
        # 不轉換
        image_to_show = image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)
    
    # 建立三個並排的子圖
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    
    # 顯示原始影像
    axs[0].imshow(image_to_show)
    axs[0].set_title("Input Image")
    axs[0].axis("off")
    
    # 顯示疊加了 GT Mask 的影像
    axs[1].imshow(gt_overlay)
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis("off")
    
    # 顯示疊加了 Predicted Mask 的影像
    axs[2].imshow(pred_overlay)
    axs[2].set_title("Predicted Mask")
    axs[2].axis("off")
    
    plt.tight_layout()
    plt.savefig("mask_compare.png", dpi=300)
    plt.show()

# # 示例测试代码
# if __name__ == '__main__':
#     # 模拟训练过程数据
#     epochs = np.arange(1, 11)
#     train_losses = np.random.uniform(0.5, 1.0, size=10)
#     val_losses = np.random.uniform(0.4, 0.9, size=10)
#     dice_scores = np.random.uniform(0.6, 0.95, size=10)
#     plot_training_metrics(epochs, train_losses, val_losses, dice_scores)

#     # 测试 detect 物体的可视化
#     # 创建一张白色背景图
#     image = np.ones((256, 256, 3), dtype=np.uint8) * 255
#     # 创建一个空 mask，并绘制两个圆形区域（模拟两个物体）
#     mask = np.zeros((256, 256), dtype=np.uint8)
#     cv2.circle(mask, (80, 80), 30, 1, -1)     # 物体1
#     cv2.circle(mask, (180, 180), 40, 1, -1)     # 物体2
#     result_image = visualize_detections(image, mask, alpha=0.5)
    
#     plt.figure(figsize=(6, 6))
#     plt.imshow(result_image)
#     plt.title("Detected Objects Visualization")
#     plt.axis('off')
#     plt.show()