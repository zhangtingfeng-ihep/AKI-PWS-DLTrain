import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import jaccard_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import cv2
import gc
import tifffile
from torch.nn import functional as F
from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration parameters
SLICE_DIR = r"F:\Sample9\Slice"
LABEL_DIR = r"F:\Sample9\Label"
OUTPUT_DIR = r"F:\Sample9\Swin-Unet\Model_Output4"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PATCH_SIZE = 256
PATCHES_PER_IMAGE = 32
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 3
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

ROTATION_RANGE = 10
ZOOM_RANGE = 0.1
SHEAR_RANGE = 5
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True

MIN_IMPROVEMENT = 0.001
PATIENCE = 5

# Swin-Unet configuration
class Config:
    class DATA:
        IMG_SIZE = 256
    class MODEL:
        class SWIN:
            PATCH_SIZE = 4
            IN_CHANS = 1
            EMBED_DIM = 96
            DEPTHS = [2, 2, 6, 2]
            NUM_HEADS = [3, 6, 12, 24]
            WINDOW_SIZE = 8
            MLP_RATIO = 4.0
            QKV_BIAS = True
            QK_SCALE = None
            DROP_RATE = 0.0
            DROP_PATH_RATE = 0.1
            APE = False
            PATCH_NORM = True
    class TRAIN:
        USE_CHECKPOINT = False

config = Config()

# Swin-Unet model
class SwinUNet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=NUM_CLASSES):
        super(SwinUNet, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=self.num_classes,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.SWIN.DROP_RATE,
            drop_path_rate=config.MODEL.SWIN.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )

    def forward(self, x):
        logits = self.swin_unet(x)
        return logits

# Load and preprocess data
def load_and_preprocess_data():
    print("Loading file lists...")
    slice_files = sorted([os.path.join(SLICE_DIR, f) for f in os.listdir(SLICE_DIR) if f.endswith('.tiff')])
    label_files = sorted([os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) if f.endswith('.tif')])
    assert len(slice_files) == len(label_files), "Mismatch in number of slice and label files"
    print(f"Found {len(slice_files)} pairs of slice and label files")
    return slice_files, label_files

# Standardize image
def standardize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std if std > 0 else image - mean

# Data augmentation
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
    mask = np.clip(mask, 1, 3)
    return patch, mask

# Extract patches
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

# Custom dataset
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
                mask = tifffile.imread(label_file).astype(np.uint8) - 1
                vessel_ratio = np.sum(mask == 2) / mask.size
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
        mask_tensor = torch.from_numpy(mask_patch).long() - 1
        assert mask_tensor.min() >= 0 and mask_tensor.max() <= 2, f"Label out of range: min={mask_tensor.min()}, max={mask_tensor.max()}"
        return patch_tensor, mask_tensor

# Early stopping mechanism
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
                print(f"Validation loss decreased, saving model to {path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered, stopping training")

# Train model
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=EPOCHS):
    print("Starting model training...")
    checkpoint_path = os.path.join(OUTPUT_DIR, 'model_checkpoint.pth')
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_IMPROVEMENT)
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'val_iou': [], 'val_recall': [], 'val_f1': [], 'val_dice': [],
        'val_class0_acc': [], 'val_class1_acc': [], 'val_class2_acc': [],
        'val_class0_iou': [], 'val_class1_iou': [], 'val_class2_iou': [],
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
        history['val_class2_acc'].append(val_class_acc[2])
        history['val_class0_iou'].append(val_class_iou[0])
        history['val_class1_iou'].append(val_class_iou[1])
        history['val_class2_iou'].append(val_class_iou[2])
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Val IoU: {val_iou:.4f}, Val Recall: {val_recall:.4f}, "
              f"Val F1: {val_f1:.4f}, Val Dice: {val_dice:.4f}")
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current Learning Rate: {current_lr:.6f}")
        history['learning_rate'].append(current_lr)
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print("Early stopping triggered, stopping training")
            break
    model.load_state_dict(torch.load(checkpoint_path))
    return history, model

# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_weight = dice_weight

    def forward(self, outputs, targets):
        ce = self.ce_loss(outputs, targets)
        vessel_pred = torch.softmax(outputs, dim=1)[:, 2, :, :]
        vessel_target = (targets == 2).float()
        intersection = (vessel_pred * vessel_target).sum()
        dice = 1 - (2. * intersection + 1) / (vessel_pred.sum() + vessel_target.sum() + 1)
        return ce + self.dice_weight * dice

# Train Swin-Unet model
def train_unet_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
        mask = tifffile.imread(label_file).astype(np.uint8) - 1
        for c in range(NUM_CLASSES):
            class_counts[c] += np.sum(mask == c)
        total_pixels += mask.size
    class_frequencies = class_counts / total_pixels
    class_weights = 1.0 / (class_frequencies + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Computed class weights: {class_weights.tolist()}")
    train_dataset = KidneySegmentationDataset(train_slice_files, train_label_files, augment=True, oversample_vessel=True)
    val_dataset = KidneySegmentationDataset(val_slice_files, val_label_files, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model = SwinUNet(config=config, img_size=PATCH_SIZE, num_classes=NUM_CLASSES)
    criterion = CombinedLoss(class_weights, dice_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history, model = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=EPOCHS)
    np.save(os.path.join(OUTPUT_DIR, 'training_history.npy'), history)
    with open(os.path.join(OUTPUT_DIR, 'metrics_history.txt'), 'w', encoding='utf-8') as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss\tTrain_Acc\tVal_Acc\tVal_IoU\tVal_Recall\tVal_F1\tVal_Dice\tVal_Class0_Acc\tVal_Class1_Acc\tVal_Class2_Acc\tVal_Class0_IoU\tVal_Class1_IoU\tVal_Class2_IoU\tLearning_Rate\n")
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
                    f"{history['val_class2_acc'][epoch]:.4f}\t"
                    f"{history['val_class0_iou'][epoch]:.4f}\t"
                    f"{history['val_class1_iou'][epoch]:.4f}\t"
                    f"{history['val_class2_iou'][epoch]:.4f}\t"
                    f"{history['learning_rate'][epoch]:.6f}\n")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate Change')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()
    print("Model training completed! Best model saved.")
    return model, test_slice_files, test_label_files

# Evaluate model
def evaluate_model(model, test_slice_files, test_label_files, device):
    print("Evaluating model on test set...")
    test_dataset = KidneySegmentationDataset(test_slice_files, test_label_files, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model.eval()
    model = model.to(device)
    class_weights = torch.tensor([1.0, 0.3, 10.0]).to(device)
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
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    with open(os.path.join(OUTPUT_DIR, 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {avg_test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
    return avg_test_loss, test_acc

# Compute metrics
def compute_metrics(pred, target, num_classes=3):
    pred_flat = pred.flatten().cpu().numpy()
    target_flat = target.flatten().cpu().numpy()
    iou = jaccard_score(target_flat, pred_flat, average='macro', labels=[0, 1, 2], zero_division=0)
    class_iou = []
    for c in range(num_classes):
        iou_c = jaccard_score(target_flat, pred_flat, average=None, labels=[c], zero_division=0)[0] if (target_flat == c).sum() > 0 else 0.0
        class_iou.append(iou_c)
    recall = recall_score(target_flat, pred_flat, average='macro', labels=[0, 1, 2], zero_division=0)
    f1 = f1_score(target_flat, pred_flat, average='macro', labels=[0, 1, 2], zero_division=0)
    dice = 2 * iou / (1 + iou)
    class_acc = []
    for c in range(num_classes):
        correct = ((pred_flat == c) & (target_flat == c)).sum()
        total = (target_flat == c).sum()
        class_acc.append(correct / total if total > 0 else 0.0)
    return iou, recall, f1, dice, class_acc, class_iou

# Visualize predictions
def visualize_predictions(model, test_slice_files, test_label_files, device, num_samples=3):
    print("Visualizing model predictions...")
    model.eval()
    model = model.to(device)
    indices = np.random.choice(len(test_slice_files), num_samples, replace=False)
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 5 * num_samples))
    for i, idx in enumerate(indices):
        img = tifffile.imread(test_slice_files[idx]).astype(np.float32)
        mask = tifffile.imread(test_label_files[idx]).astype(np.uint8)
        img_std = standardize_image(img)
        h, w = img.shape
        center_y = h // 2 - PATCH_SIZE // 2
        center_x = w // 2 - PATCH_SIZE // 2
        center_y = max(0, min(center_y, h - PATCH_SIZE))
        center_x = max(0, min(center_x, w - PATCH_SIZE))
        patch = img_std[center_y:center_y + PATCH_SIZE, center_x:center_x + PATCH_SIZE].copy()
        mask_patch = mask[center_y:center_y + PATCH_SIZE, center_x:center_x + PATCH_SIZE].copy()
        patch_tensor = torch.from_numpy(patch.reshape(1, 1, PATCH_SIZE, PATCH_SIZE)).float().to(device)
        with torch.no_grad():
            pred = model(patch_tensor)
            pred_mask = torch.argmax(pred, dim=1).cpu().numpy()[0]
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.title(f"Original Image {i + 1}")
        plt.imshow(patch, cmap='gray')
        plt.axis('off')
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.title(f"True Mask {i + 1}")
        plt.imshow(mask_patch, cmap='viridis')
        plt.axis('off')
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.title(f"Predicted Mask {i + 1}")
        plt.imshow(pred_mask, cmap='viridis')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_visualization.png'))
    plt.close()
    print("Prediction visualization completed, image saved.")

# Main execution
if __name__ == "__main__":
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Available GPUs: {torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
            print("Warning: No GPU detected, using CPU for training!")
        model, test_slice_files, test_label_files = train_unet_model()
        test_loss, test_acc = evaluate_model(model, test_slice_files, test_label_files, device)
        visualize_predictions(model, test_slice_files, test_label_files, device)
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pth'))
        torch.save(model, os.path.join(OUTPUT_DIR, 'final_model_full.pth'))
        print("Final model saved to:", os.path.join(OUTPUT_DIR, 'final_model.pth'))
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
