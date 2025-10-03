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
from torch.nn import functional as F

# MedT related modules
import math
from functools import partial

# Define qkv_transform
class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

# Define AxialAttention_dynamic
class AxialAttention_dynamic(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_dynamic, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        self.f_qr = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.f_kr = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.f_sve = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.f_sv = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)

        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class AxialAttention_wopos(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_wopos, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups)

        self.bn_output = nn.BatchNorm1d(out_planes * 1)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = self.bn_similarity(qk).view(N * W, 1, self.groups, H, H).sum(dim=1).view(N * W,
                                                                                                      self.groups, H, H)

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        sv = sv.reshape(N * W, self.out_planes, H)
        output = self.bn_output(sv).view(N, W, self.out_planes, 1, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))

class AxialBlock_dynamic(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_dynamic, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                                  width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AxialBlock_wopos(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_wopos, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                                width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class medt_net(nn.Module):
    def __init__(self, block, block_2, layers, num_classes, groups=8, s=0.125, img_size=32, imgchan=1):
        super(medt_net, self).__init__()
        self.inplanes = int(128 * s)
        self.dilation = 1
        self.block_size = img_size // 4

        self.conv1 = nn.Conv2d(imgchan, int(32 * s), kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(int(32 * s))
        self.conv2 = nn.Conv2d(int(32 * s), int(64 * s), kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(int(64 * s))
        self.conv3 = nn.Conv2d(int(64 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(int(128 * s))
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.encoder1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size // 2))
        self.inplanes = int(128 * s * block.expansion)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.encoder2 = self._make_layer(block, int(256 * s), layers[1], kernel_size=(img_size // 4))

        self.inplanes = int(128 * s)
        self.conv1p = nn.Conv2d(imgchan, int(32 * s), kernel_size=3, stride=1, padding=1)
        self.bn1p = nn.BatchNorm2d(int(32 * s))
        self.conv2p = nn.Conv2d(int(32 * s), int(64 * s), kernel_size=3, stride=1, padding=1)
        self.bn2p = nn.BatchNorm2d(int(64 * s))
        self.conv3p = nn.Conv2d(int(64 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.bn3p = nn.BatchNorm2d(int(128 * s))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool_p = nn.MaxPool2d(2, stride=2)

        img_size_p = img_size // 4

        self.encoder1p = self._make_layer(block_2, int(128 * s), layers[0], kernel_size=(img_size_p // 2))
        self.inplanes = int(128 * s * block_2.expansion)
        self.maxpool_p2 = nn.MaxPool2d(2, stride=2)
        self.encoder2p = self._make_layer(block_2, int(256 * s), layers[1], kernel_size=(img_size_p // 4))
        self.inplanes = int(256 * s * block_2.expansion)
        self.maxpool_p3 = nn.MaxPool2d(2, stride=2)
        self.encoder3p = self._make_layer(block_2, int(512 * s), layers[2], kernel_size=(img_size_p // 8))

        self.decoder2_p = nn.Conv2d(int(512 * s * block_2.expansion), int(256 * s * block_2.expansion), kernel_size=3,
                                    stride=1, padding=1)
        self.decoder3_p = nn.Conv2d(int(256 * s * block_2.expansion), int(128 * s * block_2.expansion), kernel_size=3,
                                    stride=1, padding=1)
        self.decoder4_p = nn.Conv2d(int(128 * s * block_2.expansion), int(128 * s * block_2.expansion), kernel_size=3,
                                    stride=1, padding=1)
        self.decoder5_p = nn.Conv2d(int(128 * s * block_2.expansion), int(128 * s * block_2.expansion), kernel_size=3,
                                    stride=1, padding=1)

        self.decoderf = nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1)
        self.adjust_p = nn.Conv2d(int(128 * s * block_2.expansion), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft_p = nn.Softmax(dim=1)

        self.decoder5 = nn.Conv2d(int(256 * s * block.expansion), int(128 * s * block.expansion), kernel_size=3,
                                  stride=1, padding=1)
        self.adjust = nn.Conv2d(int(128 * s * block.expansion), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, blocks, kernel_size=16, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=8,
                            base_width=64, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=8,
                                base_width=64, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        xin = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x1 = self.encoder1(x)
        x = self.maxpool2(x1)
        x2 = self.encoder2(x)

        x = F.relu(F.interpolate(self.decoder5(x2), scale_factor=(2, 2), mode='bilinear'))
        x = torch.add(x, x1)
        x = F.relu(F.interpolate(self.adjust(x), scale_factor=(2, 2), mode='bilinear'))

        x_loc = x.clone()

        for i in range(0, 4):
            for j in range(0, 4):
                x_p = xin[:, :, self.block_size * i:self.block_size * (i + 1), self.block_size * j:self.block_size * (j + 1)]
                x_p = self.conv1p(x_p)
                x_p = self.bn1p(x_p)
                x_p = self.relu(x_p)

                x_p = self.conv2p(x_p)
                x_p = self.bn2p(x_p)
                x_p = self.relu(x_p)
                x_p = self.conv3p(x_p)
                x_p = self.bn3p(x_p)
                x_p = self.relu(x_p)

                x_p = self.maxpool_p(x_p)

                x1_p = self.encoder1p(x_p)
                x_p = self.maxpool_p2(x1_p)
                x2_p = self.encoder2p(x_p)
                x_p = self.maxpool_p3(x2_p)
                x3_p = self.encoder3p(x_p)

                x_p = F.relu(F.interpolate(self.decoder2_p(x3_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = F.relu(F.interpolate(x_p, scale_factor=(2, 2), mode='bilinear'))
                x2_p = F.interpolate(x2_p, scale_factor=(2, 2), mode='bilinear')
                x_p = torch.add(x_p, x2_p)
                x_p = F.relu(F.interpolate(self.decoder3_p(x_p), scale_factor=(2, 2), mode='bilinear'))
                x1_p = F.interpolate(x1_p, scale_factor=(2, 2), mode='bilinear')
                x_p = torch.add(x_p, x1_p)
                x_p = F.relu(F.interpolate(self.decoder4_p(x_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = F.relu(F.interpolate(self.decoder5_p(x_p), scale_factor=(0.5, 0.5), mode='bilinear'))

                x_p = self.adjust_p(x_p)
                x_p = F.interpolate(x_p, size=(self.block_size, self.block_size), mode='bilinear', align_corners=False)
                x_loc[:, :, self.block_size * i:self.block_size * (i + 1), self.block_size * j:self.block_size * (j + 1)] = x_p

        x = torch.add(x, x_loc)
        x = F.relu(self.decoderf(x))

        return x

    def forward(self, x):
        return self._forward_impl(x)

def MedT(img_size=32, imgchan=1, num_classes=2):
    model = medt_net(AxialBlock_dynamic, AxialBlock_wopos, [1, 2, 4], num_classes=num_classes, s=0.125,
                     img_size=img_size, imgchan=imgchan)
    model.adjust_p = nn.Conv2d(int(128 * 0.125 * 2), num_classes, kernel_size=1, stride=1, padding=0)
    model.soft_p = nn.Softmax(dim=1)
    return model

# Set random seed
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
OUTPUT_DIR = r"F:\Sample9\PythonProject\PWS\MedT\Model_Output2"
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

# Image standardization function
def standardize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        return (image - mean) / std
    else:
        return image - mean

# Data augmentation function
def augment_patch(patch, mask, seed_val):
    random.seed(seed_val)
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

# Patch extraction function
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

# Data loading function
def load_and_preprocess_data():
    print("Loading file lists...")
    slice_files = sorted([os.path.join(SLICE_DIR, f) for f in os.listdir(SLICE_DIR) if f.endswith('.tif')])
    label_files = sorted([os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) if f.endswith('.tif')])
    assert len(slice_files) == len(label_files), "Number of slice and label files do not match"
    print(f"Found {len(slice_files)} pairs of slice and label files")
    return slice_files, label_files

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
                print(f'Validation loss improved, saving model to {path}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'Early stopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered, stopping training')

# Loss function
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

# Metrics computation function
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

# Model training function
def train_medt_model():
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
    print(f"Computed class weights: {class_weights.tolist()}")

    train_dataset = KidneySegmentationDataset(train_slice_files, train_label_files, augment=True, oversample_vessel=True)
    val_dataset = KidneySegmentationDataset(val_slice_files, val_label_files, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = MedT(img_size=PATCH_SIZE, imgchan=1, num_classes=NUM_CLASSES)
    criterion = CombinedLoss(class_weights, dice_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model = model.to(device)

    checkpoint_path = os.path.join(OUTPUT_DIR, 'model_checkpoint.pth')
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_IMPROVEMENT)
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'val_iou': [], 'val_recall': [], 'val_f1': [], 'val_dice': [],
        'val_class0_acc': [], 'val_class1_acc': [],
        'val_class0_iou': [], 'val_class1_iou': [],
        'learning_rate': []
    }
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        print(f"Epoch {epoch + 1}/{EPOCHS}")

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

        print(f"Epoch {epoch + 1}/{EPOCHS}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Val IoU: {val_iou:.4f}, Val Recall: {val_recall:.4f}, "
              f"Val F1: {val_f1:.4f}, Val Dice: {val_dice:.4f}, "
              f"Class 0 Acc: {val_class_acc[0]:.4f}, Class 1 Acc: {val_class_acc[1]:.4f}, "
              f"Val Class0 IoU: {val_class_iou[0]:.4f}, Val Class1 IoU: {val_class_iou[1]:.4f}")

        current_lr = scheduler.get_last_lr()[0]
        print(f"Current Learning Rate: {current_lr:.6f}")
        history['learning_rate'].append(current_lr)

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print("Early stopping triggered, stopping training")
            break

    model.load_state_dict(torch.load(checkpoint_path))
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

# Model evaluation function
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

# Prediction visualization function
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
            pred = model(patch_tensor)
            pred_mask = torch.argmax(pred, dim=1).cpu().numpy()[0]
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.title(f"Original Image {i + 1}")
        plt.imshow(patch, cmap='gray')
        plt.axis('off')
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.title(f"Ground Truth Mask {i + 1}")
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

# Main function
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

        model, test_slice_files, test_label_files = train_medt_model()
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