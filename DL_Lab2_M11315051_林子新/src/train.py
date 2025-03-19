import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# 假設你已經實作了 load_dataset 函數，從 oxford_pet.py 中導入
from oxford_pet import load_dataset
# 假設 UNet 定義在 models/unet.py 中
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
# 從 evaluate.py 中導入 evaluate 函數
from evaluate import evaluate

from utils import plot_training_metrics

def train(args):
    # 設置運行設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 載入訓練資料集
    print("載入訓練資料集...")
    train_dataset = load_dataset(args.data_path, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 載入驗證資料集
    print("載入驗證資料集...")
    val_dataset = load_dataset(args.data_path, mode="valid")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print("Train samples:", len(train_dataset))
    print("Valid samples:", len(val_dataset))

    # 建立模型，設定輸入通道為 3，輸出通道為 1（二分類分割），並移至 device
    model = UNet(in_channels=3, out_channels=1).to(device)
    #model = ResNet34_UNet(num_classes=1, bilinear=True).to(device)
    # 定義損失函數，這裡採用 BCEWithLogitsLoss（請注意模型輸出未經 sigmoid 處理）
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 用於保存最佳模型
    best_dice = 0.0
    model_save_path = os.path.join("../saved_models", "unet_model_best.pth")
    os.makedirs("../saved_models", exist_ok=True)
    
    # 用於記錄各個 Epoch 的指標
    epochs_list = []
    train_losses_list = []
    val_losses_list = []
    train_dices_list = []
    val_dices_list = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Batches", leave=False):
            # 取得輸入圖片與對應 mask
            if torch.is_tensor(batch['image']):
                images = batch['image'].clone().detach().float().to(device)
            else:
                images = torch.tensor(batch['image'], dtype=torch.float32).to(device)
            if torch.is_tensor(batch['mask']):
                masks = batch['mask'].clone().detach().float().to(device)
            else:
                masks = torch.tensor(batch['mask'], dtype=torch.float32).to(device)
            if images.max() > 1.0:
                images = images / 255.0
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
        
        epoch_loss /= len(train_loader.dataset)
        print(f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {epoch_loss:.4f}")
        
        # 計算驗證損失
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if torch.is_tensor(batch['image']):
                    images_val = batch['image'].clone().detach().float().to(device)
                else:
                    images_val = torch.tensor(batch['image'], dtype=torch.float32).to(device)
                if torch.is_tensor(batch['mask']):
                    masks_val = batch['mask'].clone().detach().float().to(device)
                else:
                    masks_val = torch.tensor(batch['mask'], dtype=torch.float32).to(device)
                if images_val.max() > 1.0:
                    images_val = images_val / 255.0
                outputs_val = model(images_val)
                loss_val = criterion(outputs_val, masks_val)
                val_loss_total += loss_val.item() * images_val.size(0)
        val_loss = val_loss_total / len(val_loader.dataset)
        print(f"[Epoch {epoch+1}/{args.epochs}] Validation Loss: {val_loss:.4f}")
        
        # 分別計算訓練與驗證的 Dice Score（使用 evaluate 函數）
        train_dice = evaluate(model, train_loader, device)
        print(f"[Epoch {epoch+1}/{args.epochs}] Train Dice: {train_dice:.4f}")
        val_dice = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch+1}/{args.epochs}] Val Dice: {val_dice:.4f}")

        # 保存指標
        epochs_list.append(epoch+1)
        train_losses_list.append(epoch_loss)
        val_losses_list.append(val_loss)
        train_dices_list.append(train_dice)
        val_dices_list.append(val_dice)

        # 若驗證 Dice Score 提升，則保存模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with Dice {best_dice:.4f} -> {model_save_path}")
    
    print("訓練結束！最佳模型已保存。")
    # 繪製並保存圖形：左圖為 Loss 隨 Epoch 變化，右圖為 Dice Score 隨 Epoch 變化
    plot_training_metrics(epochs_list, train_losses_list, val_losses_list, train_dices_list, val_dices_list)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default="../dataset/oxford-iiit-pet", help='Path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='Learning rate')
    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)
