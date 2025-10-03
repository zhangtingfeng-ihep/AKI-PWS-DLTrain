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
from torchvision.models import resnet34
from functools import partial

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
SLICE_DIR = r"F:\Sample9\PythonProject\PWS\image"
LABEL_DIR = r"F:\Sample9\PythonProject\PWS\label"
OUTPUT_DIR = r"F:\Sample9\PythonProject\PWS\Transfuse\Model_Output2"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PATCH_SIZE = 32
PATCHES_PER_IMAGE = 32
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 2
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

# Define TransFuse helper classes
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.bn = nn.BatchNorm2d(out_dim) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        self.need_skip = inp_dim != out_dim

    def forward(self, x):
        residual = self.skip_layer(x) if self.need_skip else x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out

class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)
        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x, bp], 1))
        if self.drop_rate > 0:
            return self.dropout(fuse)
        return fuse

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))

class Up(nn.Module):
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)
        self.attn_block = None

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

# Define DeiT model
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DeiT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4, qkv_bias=True, drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        pe = self.pos_embed[:, 1:, :]
        x = x + pe
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

def deit_small_patch16_224(pretrained=False, img_size=224, **kwargs):
    model = DeiT(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    if pretrained:
        ckpt = torch.load('pretrained/deit_small_patch16_224-cd65a155.pth')
        model.load_state_dict(ckpt['model'], strict=False)
    return model

# Define TransFuse model
class TransFuse_S(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0.2, pretrained=False):
        super(TransFuse_S, self).__init__()
        if pretrained:
            self.resnet = resnet34(weights='DEFAULT')
        else:
            self.resnet = resnet34(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()
        self.transformer = deit_small_patch16_224(pretrained=pretrained, in_chans=1, img_size=PATCH_SIZE)
        self.up1 = Up(in_ch1=384, out_ch=128)
        self.up2 = Up(128, 64)
        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.up_c = BiFusion_block(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate/2)
        self.up_c_1_1 = BiFusion_block(ch_1=128, ch_2=128, r_2=2, ch_int=128, ch_out=128, drop_rate=drop_rate/2)
        self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128)
        self.up_c_2_1 = BiFusion_block(ch_1=64, ch_2=64, r_2=1, ch_int=64, ch_out=64, drop_rate=drop_rate/2)
        self.up_c_2_2 = Up(128, 64, 64)
        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, imgs):
        x_b = self.transformer(imgs)
        x_b = torch.transpose(x_b, 1, 2)
        spatial_size = PATCH_SIZE // 16
        x_b = x_b.view(x_b.shape[0], -1, spatial_size, spatial_size)
        x_b = self.drop(x_b)
        x_b_1 = self.up1(x_b)
        x_b_1 = self.drop(x_b_1)
        x_b_2 = self.up2(x_b_1)
        x_b_2 = self.drop(x_b_2)
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)
        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)
        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)
        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)
        x_c = self.up_c(x_u, x_b)
        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)
        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)
        map_x = F.interpolate(self.final_x(x_c), size=(PATCH_SIZE, PATCH_SIZE), mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), size=(PATCH_SIZE, PATCH_SIZE), mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), size=(PATCH_SIZE, PATCH_SIZE), mode='bilinear', align_corners=True)
        return map_x, map_1, map_2

# Load and preprocess data
def load_and_preprocess_data():
    print("Loading file lists...")
    slice_files = sorted([os.path.join(SLICE_DIR, f) for f in os.listdir(SLICE_DIR) if f.endswith('.tif')])
    label_files = sorted([os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) if f.endswith('.tif')])
    assert len(slice_files) == len(label_files), "Mismatch in number of slice and label files"
    print(f"Found {len(slice_files)} pairs of slice and label files")
    return slice_files, label_files

# Standardize image
def standardize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        return (image - mean) / std
    return image - mean

# Augment patch
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
    mask = np.clip(mask, 0, 1)
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

# Custom dataset class
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
                mask = tifffile.imread(label_file).astype(np.uint8)
                vessel_ratio = np.sum(mask == 1) / mask.size
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
        mask_tensor = torch.from_numpy(mask_patch).long()
        assert mask_tensor.min() >= 0 and mask_tensor.max() <= 1, f"Label values out of range: min={mask_tensor.min()}, max={mask_tensor.max()}"
        return patch_tensor, mask_tensor

# Early stopping class
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
                print(f'Validation loss decreased, saving model to {path}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'Early stopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered, stopping training')

# Train model
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=EPOCHS):
    print("Starting model training...")
    checkpoint_path = os.path.join(OUTPUT_DIR, 'model_checkpoint.pth')
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_IMPROVEMENT)
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'val_iou': [], 'val_recall': [], 'val_f1': [], 'val_dice': [],
        'val_class0_acc': [], 'val_class1_acc': [],
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
            map_x, map_1, map_2 = model(inputs)
            loss = criterion(map_x, targets) + 0.5 * criterion(map_1, targets) + 0.5 * criterion(map_2, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(map_x, 1)
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
                map_x, map_1, map_2 = model(inputs)
                loss = criterion(map_x, targets) + 0.5 * criterion(map_1, targets) + 0.5 * criterion(map_2, targets)
                val_loss += loss.item()
                _, predicted = torch.max(map_x, 1)
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
              f"Class 0 IoU: {val_class_iou[0]:.4f}, Class 1 IoU: {val_class_iou[1]:.4f}")

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

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_weight = dice_weight

    def forward(self, outputs, targets):
        ce = self.ce_loss(outputs, targets)
        vessel_pred = torch.softmax(outputs, dim=1)[:, 1, :, :]
        vessel_target = (targets == 1).float()
        intersection = (vessel_pred * vessel_target).sum()
        dice = 1 - (2. * intersection + 1) / (vessel_pred.sum() + vessel_target.sum() + 1)
        return ce + self.dice_weight * dice

# Main function to train TransFuse model
def train_transfuse_model():
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
        mask = tifffile.imread(label_file).astype(np.uint8)
        for c in range(NUM_CLASSES):
            class_counts[c] += np.sum(mask == c)
        total_pixels += mask.size
    class_frequencies = class_counts / total_pixels
    class_weights = 1.0 / (class_frequencies + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Calculated class weights: {class_weights.tolist()}")
    train_dataset = KidneySegmentationDataset(train_slice_files, train_label_files, augment=True, oversample_vessel=True)
    val_dataset = KidneySegmentationDataset(val_slice_files, val_label_files, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model = TransFuse_S(num_classes=NUM_CLASSES, drop_rate=0.2, pretrained=False)
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
    class_weights = torch.tensor([1.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            map_x, map_1, map_2 = model(inputs)
            loss = criterion(map_x, targets) + 0.5 * criterion(map_1, targets) + 0.5 * criterion(map_2, targets)
            test_loss += loss.item()
            _, predicted = torch.max(map_x, 1)
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

# Visualize predictions
def visualize_predictions(model, test_slice_files, test_label_files, device, num_samples=3):
    print("Visualizing model predictions...")
    model.eval()
    model = model.to(device)
    indices = np.random.choice(len(test_slice_files), num_samples, replace=False)
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
            map_x, map_1, map_2 = model(patch_tensor)
            pred_mask = torch.argmax(map_x, dim=1).cpu().numpy()[0]
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

# Execute training and evaluation
if __name__ == "__main__":
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Available GPUs: {torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
            print("Warning: GPU not detected, using CPU for training, which may be slow!")
        model, test_slice_files, test_label_files = train_transfuse_model()
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