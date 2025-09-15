import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import jaccard_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
import random
import cv2
import gc
import tifffile
from torchvision import transforms
from torch.nn import functional as F

# 1. 设置随机种子以保证结果可重复性
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 2. 配置参数
SLICE_DIR = r"F:\Sample9\PythonProject\ji\image"  # 切片数据路径
LABEL_DIR = r"F:\Sample9\PythonProject\ji\label"  # 标签数据路径
OUTPUT_DIR = r"F:\Sample9\PythonProject\ji\Unet\output_model2"  # 输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 模型训练参数
PATCH_SIZE = 192
PATCHES_PER_IMAGE = 4
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 2  # 修改为2分类
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# 数据增强参数
ROTATION_RANGE = 10
ZOOM_RANGE = 0.1
SHEAR_RANGE = 5
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True

# 早停参数
MIN_IMPROVEMENT = 0.0003
PATIENCE = 10

# 3. 创建U-Net模型
class DoubleConv(nn.Module):
    """双重卷积块：(Conv2D -> ReLU) × 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """U-Net模型架构"""
    def __init__(self, in_channels=1, out_channels=NUM_CLASSES):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, NUM_CLASSES, kernel_size=1)  # 输出通道改为NUM_CLASSES=2

    def forward(self, x):
        enc1 = self.enc1(x)
        p1 = self.pool1(enc1)
        enc2 = self.enc2(p1)
        p2 = self.pool2(enc2)
        enc3 = self.enc3(p2)
        p3 = self.pool3(enc3)
        enc4 = self.enc4(p3)
        p4 = self.pool4(enc4)
        bottleneck = self.bottleneck(p4)
        bottleneck = self.dropout(bottleneck)
        up4 = self.upconv4(bottleneck)
        merge4 = torch.cat((enc4, up4), dim=1)
        dec4 = self.dec4(merge4)
        up3 = self.upconv3(dec4)
        merge3 = torch.cat((enc3, up3), dim=1)
        dec3 = self.dec3(merge3)
        up2 = self.upconv2(dec3)
        merge2 = torch.cat((enc2, up2), dim=1)
        dec2 = self.dec2(merge2)
        up1 = self.upconv1(dec2)
        merge1 = torch.cat((enc1, up1), dim=1)
        dec1 = self.dec1(merge1)
        out = self.final_conv(dec1)
        return out

# 4. 数据加载和预处理函数
def load_and_preprocess_data():
    print("开始加载文件列表...")
    slice_files = sorted([os.path.join(SLICE_DIR, f) for f in os.listdir(SLICE_DIR) if f.endswith('.tif')])
    label_files = sorted([os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) if f.endswith('.tif')])
    assert len(slice_files) == len(label_files), "切片和标签文件数量不匹配"
    print(f"找到 {len(slice_files)} 对切片和标签文件")
    return slice_files, label_files

# 5. 数据标准化函数
def standardize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        return (image - mean) / std
    else:
        return image - mean

# 6. 数据增强函数
def augment_patch(patch, mask, seed_val):
    if random.random() > 0.5:
        angle = random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
        M = cv2.getRotationMatrix2D((PATCH_SIZE / 2, PATCH_SIZE / 2), angle, 1)
        patch = cv2.warpAffine(patch, M, (PATCH_SIZE, PATCH_SIZE))
        mask = cv2.warpAffine(mask, M, (PATCH_SIZE, PATCH_SIZE))
    if random.random() > 0.5:
        scale = random.uniform(1 - ZOOM_RANGE, 1 + ZOOM_RANGE)
        M = cv2.getRotationMatrix2D((PATCH_SIZE / 2, PATCH_SIZE / 2), 0, scale)
        patch = cv2.warpAffine(patch, M, (PATCH_SIZE, PATCH_SIZE))
        mask = cv2.warpAffine(mask, M, (PATCH_SIZE, PATCH_SIZE))
    if random.random() > 0.5:
        shear = random.uniform(-SHEAR_RANGE, SHEAR_RANGE) * np.pi / 180
        M = np.float32([[1, np.tan(shear), 0], [0, 1, 0]])
        patch = cv2.warpAffine(patch, M, (PATCH_SIZE, PATCH_SIZE))
        mask = cv2.warpAffine(mask, M, (PATCH_SIZE, PATCH_SIZE))
    if HORIZONTAL_FLIP and random.random() > 0.5:
        patch = cv2.flip(patch, 1)
        mask = cv2.flip(mask, 1)
    if VERTICAL_FLIP and random.random() > 0.5:
        patch = cv2.flip(patch, 0)
        mask = cv2.flip(mask, 0)
    mask = np.round(mask).astype(np.uint8)
    mask = np.clip(mask, 0, 1)  # 修改为0-1范围
    return patch, mask

# 7. 图像块生成函数
def extract_patches(image, mask, random_patches=True):
    patches = []
    mask_patches = []
    h, w = image.shape
    if random_patches:
        for _ in range(PATCHES_PER_IMAGE):
            x = random.randint(0, w - PATCH_SIZE - 1)
            y = random.randint(0, h - PATCH_SIZE - 1)
            patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()
            mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()
            patch, mask_patch = augment_patch(patch, mask_patch, random.randint(0, 1000))
            patches.append(patch)
            mask_patches.append(mask_patch)
    else:
        step = max(1, min(h, w) // (PATCHES_PER_IMAGE ** 0.5))
        count = 0
        for y in range(0, h - PATCH_SIZE + 1, step):
            for x in range(0, w - PATCH_SIZE + 1, step):
                if count >= PATCHES_PER_IMAGE:
                    break
                patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()
                mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()
                patches.append(patch)
                mask_patches.append(mask_patch)
                count += 1
            if count >= PATCHES_PER_IMAGE:
                break
    return patches, mask_patches

# 8. 自定义数据集类
class KidneySegmentationDataset(Dataset):
    def __init__(self, slice_files, label_files, augment=True, oversample_vessel=True):
        self.slice_files = slice_files
        self.label_files = label_files
        self.augment = augment
        self.oversample_vessel = oversample_vessel
        self.total_patches = len(self.slice_files) * PATCHES_PER_IMAGE
        self.file_indices = []
        self.patch_indices = []
        for i in range(len(self.slice_files)):
            for j in range(PATCHES_PER_IMAGE):
                self.file_indices.append(i)
                self.patch_indices.append(j)
        if self.oversample_vessel:
            self.vessel_ratios = []
            for label_file in label_files:
                mask = tifffile.imread(label_file).astype(np.uint8)  # 标签已经是0-1
                vessel_ratio = np.sum(mask == 1) / mask.size  # Class1为前景
                self.vessel_ratios.append(vessel_ratio + 1e-6)
            self.vessel_ratios = np.array(self.vessel_ratios)
            self.vessel_ratios /= self.vessel_ratios.sum()

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        if self.oversample_vessel and self.augment:
            file_idx = np.random.choice(len(self.slice_files), p=self.vessel_ratios)
        else:
            file_idx = self.file_indices[idx]
        patch_idx = self.patch_indices[idx]
        img = tifffile.imread(self.slice_files[file_idx]).astype(np.float32)
        mask = tifffile.imread(self.label_files[file_idx]).astype(np.uint8)
        img = standardize_image(img)
        h, w = img.shape
        if self.augment:
            x = random.randint(0, w - PATCH_SIZE - 1)
            y = random.randint(0, h - PATCH_SIZE - 1)
            patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()
            mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()
            patch, mask_patch = augment_patch(patch, mask_patch, random.randint(0, 1000))
        else:
            step_h = max(1, (h - PATCH_SIZE) // (PATCHES_PER_IMAGE ** 0.5))
            step_w = max(1, (w - PATCH_SIZE) // (PATCHES_PER_IMAGE ** 0.5))
            y = min(h - PATCH_SIZE, int((patch_idx // int(PATCHES_PER_IMAGE ** 0.5)) * step_h))
            x = min(w - PATCH_SIZE, int((patch_idx % int(PATCHES_PER_IMAGE ** 0.5)) * step_w))
            patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()
            mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()
        patch_tensor = torch.from_numpy(patch.reshape(1, PATCH_SIZE, PATCH_SIZE))
        mask_tensor = torch.from_numpy(mask_patch).long()  # 移除-1操作
        assert mask_tensor.min() >= 0 and mask_tensor.max() <= 1, f"标签值超出范围: min={mask_tensor.min()}, max={mask_tensor.max()}"
        return patch_tensor, mask_tensor

# 9. 早停类
class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_IMPROVEMENT, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), path)
            if self.verbose:
                print(f'验证损失减小，保存模型到 {path}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'早停计数器: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('早停触发，停止训练')

# 10. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=EPOCHS):
    print("开始训练模型...")
    checkpoint_path = os.path.join(OUTPUT_DIR, 'model_checkpoint.pth')
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_IMPROVEMENT)
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'val_iou': [], 'val_recall': [], 'val_f1': [], 'val_dice': [],
        'val_class0_acc': [], 'val_class1_acc': [],  # 移除class2相关项
        'val_class0_iou': [], 'val_class1_iou': [],
        'learning_rate': []
    }
    model = model.to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.numel()
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_iou_sum = 0.0
        val_recall_sum = 0.0
        val_f1_sum = 0.0
        val_dice_sum = 0.0
        val_class_acc_sum = [0.0] * NUM_CLASSES
        val_class_iou_sum = [0.0] * NUM_CLASSES
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.numel()
                val_iou, val_recall, val_f1, val_dice, val_class_acc, val_class_iou = compute_metrics(predicted, targets, num_classes=NUM_CLASSES)
                val_iou_sum += val_iou
                val_recall_sum += val_recall
                val_f1_sum += val_f1
                val_dice_sum += val_dice
                for c in range(NUM_CLASSES):
                    val_class_acc_sum[c] += val_class_acc[c]
                    val_class_iou_sum[c] += val_class_iou[c]
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_iou = val_iou_sum / len(val_loader)
        val_recall = val_recall_sum / len(val_loader)
        val_f1 = val_f1_sum / len(val_loader)
        val_dice = val_dice_sum / len(val_loader)
        val_class_acc = [acc / len(val_loader) for acc in val_class_acc_sum]
        val_class_iou = [iou / len(val_loader) for iou in val_class_iou_sum]

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_iou'].append(val_iou)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_dice'].append(val_dice)
        history['val_class0_acc'].append(val_class_acc[0])
        history['val_class1_acc'].append(val_class_acc[1])
        history['val_class0_iou'].append(val_class_iou[0])
        history['val_class1_iou'].append(val_class_iou[1])

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Val IoU: {val_iou:.4f}, Val Recall: {val_recall:.4f}, "
              f"Val F1: {val_f1:.4f}, Val Dice: {val_dice:.4f}, "
              f"Class 0 Acc: {val_class_acc[0]:.4f}, Class 1 Acc: {val_class_acc[1]:.4f}, "
              f"Val Class0 IoU: {val_class_iou[0]:.4f}, Val Class1 IoU: {val_class_iou[1]:.4f}")

        current_lr = scheduler.get_last_lr()[0]
        print(f"Current Learning Rate: {current_lr:.6f}")
        scheduler.step(avg_val_loss)
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        early_stopping(avg_val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print("早停触发，停止训练")
            break

    model.load_state_dict(torch.load(checkpoint_path))
    return history, model

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_weight = dice_weight

    def forward(self, outputs, targets):
        ce = self.ce_loss(outputs, targets)
        foreground_pred = torch.softmax(outputs, dim=1)[:, 1, :, :]  # 修改为Class1（前景）
        foreground_target = (targets == 1).float()
        intersection = (foreground_pred * foreground_target).sum()
        dice = 1 - (2. * intersection + 1) / (foreground_pred.sum() + foreground_target.sum() + 1)
        return ce + self.dice_weight * dice

# 11. 主函数：训练模型
def train_unet_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    slice_files, label_files = load_and_preprocess_data()
    train_val_slice_files, test_slice_files, train_val_label_files, test_label_files = train_test_split(
        slice_files, label_files, test_size=TEST_SPLIT, random_state=seed
    )
    train_slice_files, val_slice_files, train_label_files, val_label_files = train_test_split(
        train_val_slice_files, train_val_label_files, test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT), random_state=seed
    )
    total_pixels = 0
    class_counts = np.zeros(NUM_CLASSES)
    for label_file in train_label_files:
        mask = tifffile.imread(label_file).astype(np.uint8)
        for c in range(NUM_CLASSES):
            class_counts[c] += np.sum(mask == c)
        total_pixels += mask.size
    class_frequencies = class_counts / total_pixels
    class_weights = 1.0 / (class_frequencies + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"动态计算的类别权重: {class_weights.tolist()}")

    train_dataset = KidneySegmentationDataset(train_slice_files, train_label_files, augment=True, oversample_vessel=True)
    val_dataset = KidneySegmentationDataset(val_slice_files, val_label_files, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model = UNet(in_channels=1, out_channels=NUM_CLASSES)
    criterion = CombinedLoss(class_weights, dice_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history, model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=EPOCHS)

    np.save(os.path.join(OUTPUT_DIR, 'training_history.npy'), history)
    with open(os.path.join(OUTPUT_DIR, 'metrics_history.txt'), 'w', encoding='utf-8') as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss\tTrain_Acc\tVal_Acc\tVal_IoU\tVal_Recall\tVal_F1\tVal_Dice\tVal_Class0_Acc\tVal_Class1_Acc\tVal_Class0_IoU\tVal_Class1_IoU\tLearning_Rate\n")
        for epoch in range(len(history['train_loss'])):
            f.write(f"{epoch+1}\t"
                    f"{history['train_loss'][epoch]:.4f}\t"
                    f"{history['val_loss'][epoch]:.4f}\t"
                    f"{history['train_acc'][epoch]:.4f}\t"
                    f"{history['val_acc'][epoch]:.4f}\t"
                    f"{history['val_iou'][epoch]:.4f}\t"
                    f"{history['val_recall'][epoch]:.4f}\t"
                    f"{history['val_f1'][epoch]:.4f}\t"
                    f"{history['val_dice'][epoch]:.4f}\t"
                    f"{history['val_class0_acc'][epoch]:.4f}\t"
                    f"{history['val_class1_acc'][epoch]:.4f}\t"
                    f"{history['val_class0_iou'][epoch]:.4f}\t"
                    f"{history['val_class1_iou'][epoch]:.4f}\t"
                    f"{history['learning_rate'][epoch]:.6f}\n")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rate'], label='学习率')
    plt.title('学习率变化')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()
    print("模型训练完成！最佳模型已保存。")
    return model, test_slice_files, test_label_files

# 12. 评估模型函数
def evaluate_model(model, test_slice_files, test_label_files, device):
    print("开始在测试集上评估模型...")
    test_dataset = KidneySegmentationDataset(test_slice_files, test_label_files, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model.eval()
    model = model.to(device)
    class_weights = torch.tensor([1.0, 1.0]).to(device)  # 修改为2分类权重
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == targets).sum().item()
            test_total += targets.numel()
    avg_test_loss = test_loss / len(test_loader)
    test_acc = test_correct / test_total
    print(f"测试集损失: {avg_test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    with open(os.path.join(OUTPUT_DIR, 'test_results.txt'), 'w') as f:
        f.write(f"测试集损失: {avg_test_loss:.4f}\n")
        f.write(f"测试集准确率: {test_acc:.4f}\n")
    return avg_test_loss, test_acc

# 13. 计算指标函数
def compute_metrics(pred, target, num_classes=2):
    pred_flat = pred.flatten().cpu().numpy()
    target_flat = target.flatten().cpu().numpy()
    iou = jaccard_score(target_flat, pred_flat, average='macro', labels=[0, 1], zero_division=0)
    class_iou = []
    for c in range(num_classes):
        iou_c = jaccard_score(target_flat, pred_flat, average=None, labels=[c], zero_division=0)[0] if (target_flat == c).sum() > 0 else 0.0
        class_iou.append(iou_c)
    recall = recall_score(target_flat, pred_flat, average='macro', labels=[0, 1], zero_division=0)
    f1 = f1_score(target_flat, pred_flat, average='macro', labels=[0, 1], zero_division=0)
    dice = 2 * iou / (1 + iou)
    class_acc = []
    for c in range(num_classes):
        correct = ((pred_flat == c) & (target_flat == c)).sum()
        total = (target_flat == c).sum()
        class_acc.append(correct / total if total > 0 else 0.0)
    return iou, recall, f1, dice, class_acc, class_iou

# 14. 预测结果可视化
def visualize_predictions(model, test_slice_files, test_label_files, device, num_samples=3):
    print("可视化模型预测结果...")
    model.eval()
    model = model.to(device)
    indices = np.random.choice(len(test_slice_files), num_samples, replace=False)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 5 * num_samples))
    for i, idx in enumerate(indices):
        img = tifffile.imread(test_slice_files[idx]).astype(np.float32)
        mask = tifffile.imread(test_label_files[idx]).astype(np.uint8)
        img_std = standardize_image(img)
        h, w = img.shape
        center_y = max(0, min(h // 2 - PATCH_SIZE // 2, h - PATCH_SIZE))
        center_x = max(0, min(w // 2 - PATCH_SIZE // 2, w - PATCH_SIZE))
        patch = img_std[center_y:center_y + PATCH_SIZE, center_x:center_x + PATCH_SIZE].copy()
        mask_patch = mask[center_y:center_y + PATCH_SIZE, center_x:center_x + PATCH_SIZE].copy()
        patch_tensor = torch.from_numpy(patch.reshape(1, 1, PATCH_SIZE, PATCH_SIZE)).float().to(device)
        with torch.no_grad():
            pred = model(patch_tensor)
            pred_mask = torch.argmax(pred, dim=1).cpu().numpy()[0]
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.title(f"原始图像 {i + 1}")
        plt.imshow(patch, cmap='gray')
        plt.axis('off')
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.title(f"真实掩码 {i + 1}")
        plt.imshow(mask_patch, cmap='viridis')
        plt.axis('off')
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.title(f"预测掩码 {i + 1}")
        plt.imshow(pred_mask, cmap='viridis')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_visualization.png'))
    plt.close()
    print("预测结果可视化完成，图像已保存。")

# 15. 执行训练和评估
if __name__ == "__main__":
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"可用GPU数量: {torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
            print("警告: 未检测到GPU，将使用CPU进行训练，这可能会很慢！")
        model, test_slice_files, test_label_files = train_unet_model()
        test_loss, test_acc = evaluate_model(model, test_slice_files, test_label_files, device)
        visualize_predictions(model, test_slice_files, test_label_files, device)
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pth'))
        torch.save(model, os.path.join(OUTPUT_DIR, 'final_model_full.pth'))
        print("最终模型已保存到:", os.path.join(OUTPUT_DIR, 'final_model.pth'))
    except Exception as e:
        import traceback
        print(f"发生错误: {e}")
        traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()