import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
from tqdm import tqdm
import timm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
import utils
from models.adaptformer import set_adapter
from models.lora import set_lora_adapter
from models.bi_adaptformer import set_bi_adapter
from models.bi_lora import set_bi_lora
from models.vptdeep_original  import vit_base_patch16_224_in21k_vptdeep
import time


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, output, label):
        self.sum += (output.argmax(dim=1).view(-1) == label.view(-1)).long().sum()
        self.count += label.size(0)

    def result(self):
        return self.sum / self.count


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


def compute_prediction_graph(logits):
    probs = F.softmax(logits, dim=-1)  # [B, C]
    f1 = probs.unsqueeze(1)  # [N, 1, C]
    f2 = probs.unsqueeze(0)  # [1, N, C]
    pred_graph = F.relu(F.cosine_similarity(f1, f2, dim=-1))  # [N, N]
    return pred_graph


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = AverageMeter()
    model = model.cuda()
    for batch in tqdm(dl):
        x, y = batch[0].cuda(), batch[1].cuda()
        out, graph = model(x)
        out = out.data
        acc.update(out, y)
    return acc.result().item()


def graph_alignment_loss(feature_graphs, pred_graph, layer_subset='all'):
    """
    feature_graphs: List[Tensor] – feature graphs from each transformer block
    pred_graph: Tensor – prediction similarity graph [B, B]
    layer_subset: str – one of 'all', 'early', or 'late'
    num_layers: int – total number of ViT layers (typically 12 or 24)
    """
    loss = 0.0

    # Determine which layers to regularize
    L = len(feature_graphs)
    if layer_subset == 'all':
        indices = range(L)
    elif layer_subset == 'early':
        indices = range(L // 2)
    elif layer_subset == 'late':
        indices = range(L // 2, L)
    else:
        raise ValueError(f"Unknown layer subset: {layer_subset}")

    for idx in indices:
        loss += F.mse_loss(feature_graphs[idx], pred_graph)

    return loss / len(indices)


def train(config, model, dl, test_dl, opt, scheduler, epoch, lambda_value, layer_subset="all"):
    model.train()
    model = model.cuda()
    for ep in tqdm(range(epoch)):
        model.train()
        model = model.cuda()
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out, feature_graphs = model(x)
            loss = F.cross_entropy(out, y)

            # regularization
            prediction_graph = compute_prediction_graph(out)
            graph_reg = graph_alignment_loss(feature_graphs, prediction_graph, layer_subset=layer_subset)
            total_loss = loss + lambda_value * graph_reg

            opt.zero_grad()
            total_loss.backward()
            opt.step()

        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            acc = test(model, test_dl)
            if acc > config['best_acc']:
                config['best_acc'] = acc
                utils.save(config['method'], config['dataset'], model, acc, ep)
    model = model.cpu()
    return model


def modify_vit_forward(vit):
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        feature_graphs = []
        for blk in self.blocks:
            x, graph = blk(x)
            feature_graphs.append(graph)

        x = self.norm(x)
        return x, feature_graphs

    def forward(self, x):
        x, feature_graphs = self.forward_features(x)
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        return logits, feature_graphs

    vit.forward_features = forward_features.__get__(vit)
    vit.forward = forward.__get__(vit)


def modify_vpt_forward(vit, feature='token_mean'):
    vit.graph_layer = GraphLayer(feature_to_use=feature)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        feature_graphs = []

        if len(self.prompt_embeddings) != len(self.blocks):
            raise RuntimeError(
                "VPT-Deep requires one prompt tensor per transformer block."
            )

        for i, blk in enumerate(self.blocks):
            prompt = self.prompt_embeddings[i].expand(B, -1, -1)
            x = torch.cat((x[:, :1], prompt, x[:, 1:]), dim=1)

            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

            x = torch.cat(
                (x[:, :1], x[:, 1 + self.prompt_len:]),
                dim=1
            )
            graph = self.graph_layer(x)
            feature_graphs.append(graph)

        x = self.norm(x)
        return x, feature_graphs

    def forward(self, x):
        x, feature_graphs = self.forward_features(x)

        if self.global_pool == 'avg':
            cls_features = x[:, self.num_tokens:].mean(dim=1)
        else:
            cls_features = x[:, 0]

        cls_features = self.fc_norm(cls_features)
        logits = self.head(cls_features)
        return logits, feature_graphs, cls_features

    vit.forward_features = forward_features.__get__(vit)
    vit.forward = forward.__get__(vit)
    return vit

def build_model_and_data(args, config):
    if args.task == 'vtab':
        train_dl, test_dl = utils.get_vtab_data(
            args.dataset,
            evaluate=True,
            normalize=False,
            batch_size=64,
            num_workers=2,
            is_hdf5=args.hdf5
        )
        classes_dim = utils.get_vtab_classes_num(args.dataset)

    elif args.task == 'fs':
        train_dl, val_dl, test_dl = utils.get_few_shot_data(
            args.dataset,
            batch_size=64,
            num_workers=4,
            shot=args.fs_shot,
            seed=args.fs_seed,
            is_hdf5=args.hdf5
        )
        classes_dim = utils.get_few_shot_classes_num(args.dataset)

    if args.method == 'vptdeep':
        vit = create_model(
           "vit_base_patch16_224_in21k_vptdeep",
            checkpoint_path='./ViT-B_16.npz',
            drop_path_rate=0.1,
            tuning_mode='vptdeep',
            prompt_len=args.prompt_len,
            insertlength=args.vpt_depth
        )
        vit.reset_classifier(classes_dim)
        modify_vpt_forward(vit, feature=args.feature)
    else:
        vit = create_model(
            args.model,
            checkpoint_path='./ViT-B_16.npz',
            drop_path_rate=0.1
        )
        vit.reset_classifier(classes_dim)
        modify_vit_forward(vit)

    
    if args.method == 'adaptformer':
        set_adapter(vit, dim=args.dim, s=config['scale'], feature=args.feature)
    elif args.method == 'lora':
        set_lora_adapter(vit, dim=args.dim, s=config['scale'], feature=args.feature)
    elif args.method == 'bi-adaptformer':
        set_bi_adapter(vit, dim=32, s=config['scale'], bit=args.bit, feature=args.feature)
    elif args.method == 'bi-lora':
        set_bi_lora(vit, dim=32, s=config['scale'], bit=args.bit, feature=args.feature)
    elif args.method == 'vptdeep':
        pass
    else:
        raise ValueError(f"Unknown method:{args.method}")
    return vit, train_dl, test_dl


def setup_optimizer(model, args, config):
    trainable = []

    for n, p in model.named_parameters():
        if args.method == 'vptdeep':
            should_train = 'prompt_embeddings' in n or n.startswith('head.')
        else:
            should_train = 'adapter' in n or 'head' in n

        if should_train:
            trainable.append(p)
            p.requires_grad = True
        else:
            p.requires_grad = False

    if not trainable:
        raise RuntimeError(f"No trainable parameters selected for {args.method}")

    if args.method == 'vptdeep':
        trainable_names = [
            n for n, p in model.named_parameters() if p.requires_grad
        ]
        prompt_names = [
            n for n in trainable_names if 'prompt_embeddings' in n
        ]
        if len(prompt_names) != len(model.blocks):
            raise RuntimeError(
                f"Expected {len(model.blocks)} trainable prompt tensors, "
                f"found {len(prompt_names)}."
            )
        unexpected = [
            n for n in trainable_names
            if 'prompt_embeddings' not in n and not n.startswith('head.')
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected trainable VPT parameters: {unexpected}")

    opt = AdamW( trainable, lr=args.lr,weight_decay=args.wd)

    scheduler = CosineLRScheduler(opt, t_initial=args.epochs, warmup_t=10,lr_min=1e-5,warmup_lr_init=1e-6 )

    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Number of trainable params: {n_parameters / 1e6:.4f}M")

    if args.method == 'vptdeep':
        prompt_parameters = sum(
            p.numel() for n, p in model.named_parameters()
            if p.requires_grad and 'prompt_embeddings' in n
        )
        print(f"Prompt parameters: {prompt_parameters / 1e6:.4f}M")
        backbone_parameters = sum(
            p.numel() for n, p in model.named_parameters()
            if not n.startswith('head.') and 'prompt_embeddings' not in n
        )
        print(
            "Prompt fraction of frozen backbone: "
            f"{100.0 * prompt_parameters / backbone_parameters:.3f}%"
        )
    return opt, scheduler


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument( '--task',type=str,default='vtab', choices=['vtab', 'fs'])
    parser.add_argument('--fs_shot', type=int, default=1)
    parser.add_argument('--fs_seed', type=int, default=0)
    parser.add_argument('--prompt_len', type=int, default=50)
    parser.add_argument('--vpt_depth', type=int, default=12)
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--method', type=str, default='adaptformer')
    parser.add_argument('--bit', type=int, default=1, choices=[1, 2, 4, 8, 32])
    parser.add_argument('--lambda_value', type=float, default=0)
    parser.add_argument('--reg_layers', type=str, default='all', choices=['all', 'early', 'late'])
    parser.add_argument('--hdf5', action='store_true', default=False)
    parser.add_argument('--feature', type=str, default='token_mean',
                       choices=['cls_token', 'token_mean', 'token_max'],
                       help='feature to compute similarity graphs')
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_seed(args.seed)

    config = utils.get_config(args.method, args.dataset)
    config["best_acc"] = 0
    config["method"] = args.method
    config["dataset"] = args.dataset

    vit, train_dl, test_dl = build_model_and_data(args, config)
    vit = vit.to(device)

    opt, scheduler = setup_optimizer(vit, args, config)
    model = train(config,vit,train_dl,test_dl,opt,scheduler, epoch=args.epochs,lambda_value=args.lambda_value,layer_subset=args.reg_layers)

    best_acc = config['best_acc']
    print(best_acc)
    