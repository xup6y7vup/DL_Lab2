import torch
import numpy as np
from utils import dice_score  # 确保 utils.py 与 evaluate.py 在同一目录或在 Python 路径中

def evaluate(net, data, device):
    """
    在给定数据集上评估模型，计算平均 Dice Score。

    参数：
    - net: 分割模型
    - data: 数据加载器（例如 torch.utils.data.DataLoader），迭代返回包含 'image' 和 'mask' 的字典
    - device: 运行设备（如 'cpu' 或 'cuda'）

    返回：
    - average_dice: 数据集中所有样本的平均 Dice Score
    """
    net.eval()  # 设置模型为评估模式
    dice_scores = []
    
    with torch.no_grad():
        for batch in data:
            # 获取图像和标签
            images = batch['image']
            gt_masks = batch['mask']
            
            # 如果 images 不是浮点类型，则转换为 float，并归一化（假设输入值在0～255之间）
            if not torch.is_floating_point(images):
                images = images.float()
            if images.max() > 1.0:
                images = images / 255.0
            
            # 对 gt_masks 同样转换为 float（视具体数据而定）
            if not torch.is_floating_point(gt_masks):
                gt_masks = gt_masks.float()
            
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            
            # 模型推理
            outputs = net(images)
            # 假设模型输出为单通道 logits，使用 sigmoid 激活后进行二值化
            if outputs.shape[1] == 1:
                outputs = torch.sigmoid(outputs)
                pred_masks = outputs > 0.5  # 布尔型
            else:
                pred_masks = outputs.argmax(dim=1)
            
            # 转换为 numpy 数组计算 Dice Score
            pred_masks_np = pred_masks.cpu().numpy()
            gt_masks_np = gt_masks.cpu().numpy()
            
            # 针对批量中每个样本计算 Dice Score
            for pred_mask, gt_mask in zip(pred_masks_np, gt_masks_np):
                dice = dice_score(pred_mask, gt_mask)
                dice_scores.append(dice)
    
    average_dice = np.mean(dice_scores)
    print("Average Dice Score: {:.4f}".format(average_dice))
    return average_dice
# # 示例用法（注意：这段代码仅供参考，需要根据你的项目实际情况调整）
# if __name__ == '__main__':
#     # 假设 net, test_loader, device 已经定义并初始化
#     # average_dice = evaluate(net, test_loader, device)
#     pass
