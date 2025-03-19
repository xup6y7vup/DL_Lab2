import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 從 utils.py 中匯入 visualize_comparison
from utils import visualize_comparison  

# 從 oxford_pet.py 中導入 load_dataset 函數
from oxford_pet import load_dataset
# 從 models/unet.py 導入 UNet 模型
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
# 從 evaluate.py 導入 evaluate 函數
from evaluate import evaluate

def get_args():
    parser = argparse.ArgumentParser(description='Inference for segmentation model')
    parser.add_argument('--model', default='../saved_models/unet_model_best.pth', help='Path to stored model weights')
    parser.add_argument('--data_path', type=str, default='../dataset/oxford-iiit-pet', help='Path to dataset')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='Batch size')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 載入模型
    net = UNet(in_channels=3, out_channels=1)
    #net = ResNet34_UNet(num_classes=1, bilinear=True)
    net.to(device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    
    # 使用 load_dataset 載入測試集（模式設為 "test"）
    test_dataset = load_dataset(args.data_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 如果測試資料中包含 GT，則計算 Dice Score
    sample = test_dataset[0]
    if sample.get('mask') is not None:
        print("Evaluating test Dice Score...")
        test_dice = evaluate(net, test_loader, device)
        print(f"Test Dice Score: {test_dice:.4f}")
    
    # 推理並顯示結果
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 只顯示前 2 個 batch，可自行調整
            if batch_idx >= 2:
                break

            # 取得影像與 (可能有的) GT Mask
            images = batch['image'].float().to(device)
            gt_masks = batch.get('mask', None)  # 可能是 tensor，也可能是 None

            # 模型推理
            outputs = net(images)
            outputs = torch.sigmoid(outputs)
            pred_masks = (outputs > 0.5).float()

            # 遍歷當前 batch 中的前 5 個樣本
            for i in range(min(3, images.size(0))):
                # 轉成 (H, W, C) 並回到 CPU
                img_tensor = images[i].cpu().numpy()
                img_np = np.transpose(img_tensor, (1, 2, 0))  # (C, H, W) -> (H, W, C)

                # 處理預測結果
                pred_mask_tensor = pred_masks[i].cpu().numpy()
                if pred_mask_tensor.shape[0] == 1:
                    pred_mask_np = pred_mask_tensor.squeeze(0)
                else:
                    pred_mask_np = pred_mask_tensor

                # 如果有 GT Mask，則取出對應的第 i 張
                if gt_masks is not None:
                    gt_mask_tensor = gt_masks[i].float().cpu().numpy()
                    if gt_mask_tensor.shape[0] == 1:
                        gt_mask_np = gt_mask_tensor.squeeze(0)
                    else:
                        gt_mask_np = gt_mask_tensor
                else:
                    # 如果沒有 GT，這裡就放 None 或直接放個全 0 mask
                    gt_mask_np = None

                # 調用 visualize_comparison
                visualize_comparison(img_np, gt_mask_np, pred_mask_np, alpha=0.5)

                # 如果想顯示檔名，可以從 batch 取 file_name（若有）
                file_name = batch.get('file_name', [f'Image_{i}'])[i] if 'file_name' in batch else f'Image_{i}'
                print(f"Displayed: {os.path.basename(file_name)}")

if __name__ == '__main__':
    main()
