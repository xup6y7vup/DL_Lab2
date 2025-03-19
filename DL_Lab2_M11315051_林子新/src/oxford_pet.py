import os
import torch
import shutil
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "val", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __len__(self):
        if self.mode == "train":
            # 每個原始樣本產生原始版本與增強版本，共2倍
            return 2 * len(self.filenames)
        else:
            return len(self.filenames)

    def __getitem__(self, idx):
        if self.mode == "train":
            original_idx = idx // 2  # 取得原始樣本索引
            sample = super().__getitem__(original_idx)
            
            # 將 numpy 陣列轉為 PIL Image 以方便後續處理
            image = Image.fromarray(sample["image"])
            mask = Image.fromarray(sample["mask"])
            trimap = Image.fromarray(sample["trimap"])
            
            if idx % 2 == 0:
                # 偶數：直接使用原始圖像
                image = image.resize((256, 256), Image.BILINEAR)
                mask = mask.resize((256, 256), Image.NEAREST)
                trimap = trimap.resize((256, 256), Image.NEAREST)
            else:
                # 奇數：進行隨機增強
                # choice = random.choice(['zoom', 'grayscale', 'colorjitter'])
                # if choice == 'zoom':
                    # 模擬放大：隨機裁剪再放大回原尺寸
                w, h = image.size
                scale = random.uniform(1.0, 1.5)
                new_w, new_h = int(w / scale), int(h / scale)
                left = random.randint(0, w - new_w)
                top = random.randint(0, h - new_h)
                image = image.crop((left, top, left + new_w, top + new_h))
                image = image.resize((w, h), Image.BILINEAR)
                mask = mask.crop((left, top, left + new_w, top + new_h))
                mask = mask.resize((w, h), Image.NEAREST)
                trimap = trimap.crop((left, top, left + new_w, top + new_h))
                trimap = trimap.resize((w, h), Image.NEAREST)
                # elif choice == 'grayscale':
                    # 轉為灰階後再轉回 RGB
                image = image.convert("L").convert("RGB")
                    # mask 與 trimap 保持不變
                # elif choice == 'colorjitter':
                    # 簡單調整亮度（可依需求加入對比、飽和度、色調調整）
                factor = random.uniform(0.7, 1.3)
                im_arr = np.array(image, dtype=np.float32)
                im_aug_arr = np.clip(im_arr * factor, 0, 255).astype(np.uint8)
                image = Image.fromarray(im_aug_arr)
                    # mask 與 trimap 保持不變

                # 增強後統一調整尺寸
                image = image.resize((256, 256), Image.BILINEAR)
                mask = mask.resize((256, 256), Image.NEAREST)
                trimap = trimap.resize((256, 256), Image.NEAREST)
            
            # 轉回 numpy 陣列並轉換格式 HWC -> CHW
            image = np.array(image)
            mask = np.array(mask)
            trimap = np.array(trimap)
            image = np.moveaxis(image, -1, 0)
            mask = np.expand_dims(mask, 0)
            trimap = np.expand_dims(trimap, 0)
            return {"image": image, "mask": mask, "trimap": trimap}
        else:
            # 驗證與測試模式直接處理，不做增強
            sample = super().__getitem__(idx)
            image = Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR)
            mask = Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST)
            trimap = Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST)
            image = np.array(image)
            mask = np.array(mask)
            trimap = np.array(trimap)
            image = np.moveaxis(image, -1, 0)
            mask = np.expand_dims(mask, 0)
            trimap = np.expand_dims(trimap, 0)
            return {"image": image, "mask": mask, "trimap": trimap}


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode):
    """
    下載資料集到 data_path 資料夾，並根據 mode ("train", "valid", "test")
    返回 SimpleOxfordPetDataset 實例，劃分比例為 90% : 10% : "test"
    """
    # 檢查是否存在 images 與 annotations 資料夾，若不存在則下載
    if not (os.path.exists(os.path.join(data_path, "images")) and 
            os.path.exists(os.path.join(data_path, "annotations"))):
        print("資料集不存在，開始下載...")
        OxfordPetDataset.download(data_path)
    else:
        print("資料集已存在，跳過下載。")
    
    # 只有訓練資料使用資料增強
    transform = None
    return SimpleOxfordPetDataset(root=data_path, mode=mode, transform=transform)


# 測試 load_dataset 與劃分效果
if __name__ == '__main__':
    data_folder = "../dataset"  # 指定下載資料夾
    # 測試不同模式下的 dataset 大小
    train_dataset = load_dataset(data_folder, "train")
    valid_dataset = load_dataset(data_folder, "valid")
    test_dataset = load_dataset(data_folder, "test")
    print("Train samples:", len(train_dataset))
    print("Valid samples:", len(valid_dataset))
    print("Test samples:", len(test_dataset))