import torch
from torch import nn
import timm
import torch.nn.functional as F


def forward_block(self, x):
    x = x + self.drop_path1(self.attn(self.norm1(x)))
    x = x + self.drop_path2(self.mlp(self.norm2(x))) + self.adapter_mlp(x) * self.s
    graph = self.graph_layer(x)
    return x, graph



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

    # FFN and adapter
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.adapter_mlp(self.norm2(x)) * self.s
    graph = self.graph_layer(x)
    return x, graph


class Adapter(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.adapter_down = nn.Linear(in_dim, dim, bias=False)
        self.adapter_up = nn.Linear(dim, in_dim, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
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


def set_adapter(model, dim=32, s=1, feature="token_mean"):
    for layer in model.children():
        if type(layer) == timm.models.vision_transformer.Block:
            hidden_dim = layer.norm2.normalized_shape[0]
            layer.adapter_mlp = Adapter(in_dim=hidden_dim, dim=dim)
            layer.s = s
            layer.graph_layer = GraphLayer(feature_to_use=feature)
            bound_method = forward_block.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif type(layer) == timm.models.swin_transformer.SwinTransformerBlock:
            layer.adapter_mlp = Adapter(in_dim=layer.dim, dim=dim)
            layer.s = s
            layer.graph_layer = GraphLayer(feature_to_use=feature)
            bound_method = forward_swin_block.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_adapter(layer, dim, s)
