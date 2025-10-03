import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair


# Helper functions for Transformer architecture
def swish(x):
    return x * torch.sigmoid(x)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# Transformer components
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config['transformer']['num_heads']
        self.attention_head_size = int(config['hidden_size'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config['hidden_size'], self.all_head_size)
        self.key = Linear(config['hidden_size'], self.all_head_size)
        self.value = Linear(config['hidden_size'], self.all_head_size)

        self.out = Linear(config['hidden_size'], config['hidden_size'])
        self.attn_dropout = Dropout(config['transformer']['attention_dropout_rate'])
        self.proj_dropout = Dropout(config['transformer']['attention_dropout_rate'])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config['hidden_size'], config['transformer']['mlp_dim'])
        self.fc2 = Linear(config['transformer']['mlp_dim'], config['hidden_size'])
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config['transformer']['dropout_rate'])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, config, in_channels=2048):
        super(Embeddings, self).__init__()
        self.config = config
        self.hybrid = None
        self.n_patches = 64
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, config['hidden_size']))
        self.dropout = Dropout(config['transformer']['dropout_rate'])

        if config['hidden_size'] != in_channels:
            self.projection = Linear(in_channels, config['hidden_size'])
        else:
            self.projection = None

    def forward(self, x):
        B, C, H, W = x.shape  # (B, 2048, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, 2048)
        if self.projection is not None:
            x = self.projection(x)  # (B, 64, 768) if hidden_size=768
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config['hidden_size']
        self.attention_norm = LayerNorm(config['hidden_size'], eps=1e-6)
        self.ffn_norm = LayerNorm(config['hidden_size'], eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config['hidden_size'], eps=1e-6)
        for _ in range(config['transformer']['num_layers']):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


# ResNet-50 Backbone for feature extraction
class ResNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetV2Block, self).__init__()
        self.conv1 = Conv2dReLU(in_channels, out_channels // 4, kernel_size=1, padding=0, use_batchnorm=True)
        self.conv2 = Conv2dReLU(out_channels // 4, out_channels // 4, kernel_size=3, padding=1,
                                stride=stride, use_batchnorm=True)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetV2(nn.Module):
    def __init__(self):
        super(ResNetV2, self).__init__()
        # Simplified ResNet-50 backbone for feature extraction
        self.in_channels = 64
        self.conv1 = Conv2dReLU(3, 64, kernel_size=7, stride=2, padding=3, use_batchnorm=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet stages
        self.layer1 = self._make_layer(64, 256, blocks=3, stride=1)
        self.layer2 = self._make_layer(256, 512, blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 1024, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 2048, blocks=3, stride=2)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetV2Block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetV2Block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []

        x = self.conv1(x)
        features.append(x)  # 1/2 scale

        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)  # 1/4 scale

        x = self.layer2(x)
        features.append(x)  # 1/8 scale

        x = self.layer3(x)
        features.append(x)  # 1/16 scale

        x = self.layer4(x)

        return x, features



class TransformerModel(nn.Module):
    def __init__(self, config, vis):
        super(TransformerModel, self).__init__()
        self.embeddings = Embeddings(config, in_channels=2048)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config['hidden_size'],
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

        decoder_channels = config['decoder_channels']
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        skip_channels = config['skip_channels']

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < len(features)) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config['classifier']
        self.img_size = img_size  # Store img_size for use in forward

        self.resnet = ResNetV2()
        self.transformer = TransformerModel(config, vis=vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = nn.Conv2d(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
            padding=1
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Get features from ResNet backbone
        x, features = self.resnet(x)

        # Pass features through Transformer
        x, attn_weights = self.transformer(x)

        # Decode and upsample with skip connections
        x = self.decoder(x, features[::-1])

        # Additional upsampling to match target size
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        # Final segmentation head
        logits = self.segmentation_head(x)

        return logits

# TransUNet configuration
def get_transunet_config(n_classes=3, patch_size=8):
    config = {}
    config['n_classes'] = n_classes
    config['patch_size'] = patch_size
    config['hidden_size'] = 768

    config['transformer'] = {
        'mlp_dim': 3072,
        'num_heads': 12,
        'num_layers': 12,
        'attention_dropout_rate': 0.0,
        'dropout_rate': 0.1
    }

    config['classifier'] = 'seg'
    config['decoder_channels'] = (256, 128, 64, 16)
    config['skip_channels'] = [1024, 512, 256, 64]
    config['n_skip'] = 3

    return config


# Main TransUNet class to use in your code
class TransUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, img_size=256):
        super(TransUNet, self).__init__()
        config = get_transunet_config(n_classes=out_channels)
        self.model = VisionTransformer(
            config,
            img_size=img_size,
            num_classes=out_channels
        )

    def forward(self, x):
        return self.model(x)