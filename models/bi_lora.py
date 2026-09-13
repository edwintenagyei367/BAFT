import torch
from torch import nn
import timm
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np


class QLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bits=1):
        super(QLinear, self).__init__(in_channels, out_channels, bias=False)
        self.q = self.quantize(bits)
        self.fake_quan = True
        self.bits = bits

    def forward(self, inputs):
        if not self.fake_quan:
            output = F.linear(inputs, self.weight, None)
            return output

        w = self.weight
        mean_w = w.mean()
        std_w = w.std()
        w = (w - mean_w) / (std_w + 1e-5)
        wb = self.q(w).data + w - w.data
        weight = wb * std_w + mean_w
        output = F.linear(inputs, weight, None)
        return output

    def dump(self):
        w = self.weight
        mean_w = w.mean()
        std_w = w.std()
        w = (w - mean_w) / (std_w + 1e-5)
        quantized = self.q(w).data.reshape(-1, 1)
        quantized = (quantized - self.code.reshape(1, -1)).abs().argmin(dim=1)
        byte_str = b''
        byte = 0
        for i in range(len(quantized)):
            if i % (8 // self.bits) == 0:
                byte = 0
            byte += quantized[i].item() * (2 ** self.bits) ** (i % (8 // self.bits))
            if i % (8 // self.bits) == (8 // self.bits) - 1:
                byte_str += byte.to_bytes(1, 'big')
        return byte_str, mean_w, std_w

    def load(self, byte_str, mean_w, std_w):
        self.fake_quan = False
        quantized = torch.zeros_like(self.weight.reshape(-1))
        for i in range(len(byte_str)):
            byte = byte_str[i]
            for j in range(8 // self.bits):
                quantized[i * (8 // self.bits) + j].data += self.code[byte % (2 ** self.bits)].data
                byte //= (2 ** self.bits)
        quantized = quantized * std_w + mean_w
        self.weight.data = quantized.reshape(*self.weight.size())

    def quantize(self, bit=1):
        j = 2 ** bit * 2
        ppf = np.array([0 for _ in range(1, j)])
        values = ppf[::2]
        ranges = ppf[1::2]
        for i in range(500):
            ranges = (values[1:] + values[:-1]) / 2
            pv = norm.cdf(ranges)
            pv = np.insert(pv, 0, 0)
            pv = np.insert(pv, len(pv), 1)
            pv = (pv[1:] + pv[:-1]) / 2
            values = norm.ppf(pv)
        value = torch.tensor(values).float()
        self.code = value
        pos = torch.tensor(ranges).float()
        delta = (value[1:] - value[:-1]) / 2

        def func(x):
            pos_ = pos.to(x.device)
            delta_ = delta.to(x.device)
            x = x.unsqueeze(dim=-1)
            x = x - pos_
            x = torch.sign(x)
            x *= delta_
            return x.sum(-1)

        return func


def forward_block(self, x):
    x = x + self.drop_path1(self.attn(self.norm1(x)))
    x = x + self.drop_path2(self.mlp(self.norm2(x)))
    graph = self.graph_layer(x)
    return x, graph


def forward_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    delta_q = self.lora_q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) * self.s
    delta_v = self.lora_v(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) * self.s
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
    q, v = q + delta_q[0], v + delta_v[0]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def forward_attn_swin(self, x, mask=None):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    delta_q = self.lora_q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) * self.s
    delta_v = self.lora_v(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) * self.s
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q + delta_q[0]
    v = v + delta_v[0]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        N, N, -1).permute(2, 0, 1).contiguous()
    attn = attn + relative_position_bias.unsqueeze(0)
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)

    attn = self.attn_drop(attn.softmax(dim=-1))
    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def forward_swin_block(self, x):
    B, L, C = x.shape
    H = W = int(L ** 0.5)
    assert L == H * W, f"Input feature has wrong size (L={L} vs H*W={H * W})"

    shortcut = x
    x = self.norm1(x)
    x = x.view(B, H, W, C)

    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x

    # partition windows
    x_windows = timm.models.swin_transformer.window_partition(shifted_x, self.window_size)
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

    # attention
    attn_windows = self.attn(x_windows, mask=self.attn_mask)

    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = timm.models.swin_transformer.window_reverse(attn_windows, self.window_size, H, W)

    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x

    x = x.view(B, H * W, C)
    # FFN
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    graph = self.graph_layer(x)
    return x, graph


class Adapter(nn.Module):
    def __init__(self, in_dim, dim, bit):
        super().__init__()

        if bit == 32:
            self.adapter_down = nn.Linear(in_dim, dim, bias=False)
            self.adapter_up = nn.Linear(dim, in_dim, bias=False)
            nn.init.zeros_(self.adapter_up.weight)
        else:
            self.adapter_down = QLinear(in_dim, dim, bit)
            self.adapter_up = QLinear(dim, in_dim, bit)
            nn.init.trunc_normal_(self.adapter_up.weight, mean=0.0, std=0.001, a=-0.002, b=0.002)
        self.act = nn.Identity()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
        return x_up


class GraphLayer(nn.Module):
    def __init__(self, feature_to_use=None):
        super().__init__()
        self.feature_store = []
        self.feature = feature_to_use

    def forward(self, x):
        if self.feature == "cls_token" and x.dim() == 3:
            x = x[:, 0, :]
        elif self.feature == 'token_mean' and x.dim() == 3:
            x = x.mean(dim=1)
        elif self.feature == 'token_max' and x.dim() == 3:
            x = x.max(dim=1).values
        f1 = x.unsqueeze(1)
        f2 = x.unsqueeze(0)
        graph = F.relu(F.cosine_similarity(f1, f2, dim=-1))  # [B, B]
        return graph


# def set_bi_lora(model, dim=32, s=1, bit=1, feature="token_mean"):
#     for name, layer in model.named_modules():
#         if isinstance(layer, timm.models.vision_transformer.Attention):
#             layer.lora_q = Adapter(dim, bit)
#             layer.lora_v = Adapter(dim, bit)
#             layer.s = s
#             # layer.graph_layer = GraphLayer()
#             bound_method = forward_attn.__get__(layer, layer.__class__)
#             setattr(layer, 'forward', bound_method)
#         elif isinstance(layer, timm.models.vision_transformer.Block):
#             layer.graph_layer = GraphLayer(feature_to_use=feature)
#             bound_method = forward_block.__get__(layer, layer.__class__)
#             setattr(layer, 'forward', bound_method)


def set_bi_lora(model, dim=32, s=1, bit=1, feature="token_mean"):
    for name, layer in model.named_modules():
        if isinstance(layer, timm.models.vision_transformer.Attention):
            hidden_dim = layer.qkv.in_features
            layer.lora_q = Adapter(in_dim=hidden_dim, dim=dim, bit=bit)
            layer.lora_v = Adapter(in_dim=hidden_dim, dim=dim, bit=bit)
            layer.s = s
            # layer.graph_layer = GraphLayer()
            bound_method = forward_attn.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif isinstance(layer, timm.models.swin_transformer.WindowAttention):
            hidden_dim = layer.qkv.in_features
            layer.lora_q = Adapter(hidden_dim, dim)
            layer.lora_v = Adapter(hidden_dim, dim)
            layer.s = s
            bound_method = forward_attn_swin.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif isinstance(layer, timm.models.swin_transformer.SwinTransformerBlock):
            layer.graph_layer = GraphLayer(feature_to_use=feature)
            bound_method = forward_swin_block.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif isinstance(layer, timm.models.vision_transformer.Block):
            layer.graph_layer = GraphLayer(feature_to_use=feature)
            bound_method = forward_block.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
