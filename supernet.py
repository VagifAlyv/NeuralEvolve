"""Weight-sharing supernet (Once-for-All / BigNAS style).

One big network contains every architecture of the search space:
- depth:      a stage with depth d uses its first d blocks
- width:      a layer with c channels uses the first c filters
- expansion:  a block uses the first (in_channels * expand) hidden channels
- kernel:     a 3x3 or 5x5 kernel is the CENTER of the shared 7x7 kernel
- SE:         the squeeze-excite module is skipped when the gene is 0
- resolution: the batch is resized before going in

Training uses the sandwich rule: every batch trains the largest, the smallest and a few random
subnets. The largest learns from the labels, the others learn from the largest (in-place distillation).
Weights are trained by SGD only; evolution never touches them.

    python supernet.py --epochs 100 --tag main
"""
import argparse
import csv
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import genome as G
from common import RUNS_DIR, get_device, seed_everything
from data import TRAIN_RES, Data, prepare


class DynBN(nn.Module):
    """BatchNorm with sliced affine parameters. Always normalizes with batch statistics; every
    extracted subnet gets its own running statistics by recalibration (see evaluate.py)."""

    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        c = x.size(1)
        return F.batch_norm(x, None, None, self.weight[:c], self.bias[:c], training=True, eps=1e-5)


class DynPointwise(nn.Module):
    """1x1 conv that uses the first `cout` filters and the first in_channels inputs."""

    def __init__(self, max_in, max_out, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(max_out, max_in, 1, 1))
        nn.init.kaiming_normal_(self.weight, mode="fan_out")
        self.bias = nn.Parameter(torch.zeros(max_out)) if bias else None

    def forward(self, x, cout):
        w = self.weight[:cout, :x.size(1)]
        return F.conv2d(x, w, None if self.bias is None else self.bias[:cout])


class DynDepthwise(nn.Module):
    """Depthwise conv with a shared 7x7 kernel; smaller kernels take its center."""

    def __init__(self, max_c, max_k=max(G.KERNELS)):
        super().__init__()
        self.max_k = max_k
        self.weight = nn.Parameter(torch.empty(max_c, 1, max_k, max_k))
        nn.init.kaiming_normal_(self.weight, mode="fan_out")

    def crop(self, c, k):
        s = (self.max_k - k) // 2
        return self.weight[:c, :, s:s + k, s:s + k]

    def forward(self, x, k, stride):
        c = x.size(1)
        return F.conv2d(x, self.crop(c, k), None, stride, k // 2, 1, c)


class DynSE(nn.Module):
    def __init__(self, max_c):
        super().__init__()
        r = G.se_channels(max_c)
        self.fc1 = DynPointwise(max_c, r, bias=True)
        self.fc2 = DynPointwise(r, max_c, bias=True)

    def forward(self, x):
        c = x.size(1)
        s = F.relu(self.fc1(x.mean((2, 3), keepdim=True), G.se_channels(c)))
        return x * F.hardsigmoid(self.fc2(s, c))


class DynMBConv(nn.Module):
    """Same structure and names as genome.MBConv, but every size is chosen at forward time."""

    def __init__(self, max_in, max_out, stride):
        super().__init__()
        max_hidden = max_in * max(G.EXPANDS)
        self.expand_conv = DynPointwise(max_in, max_hidden)
        self.expand_bn = DynBN(max_hidden)
        self.dw_conv = DynDepthwise(max_hidden)
        self.dw_bn = DynBN(max_hidden)
        self.se = DynSE(max_hidden)
        self.proj_conv = DynPointwise(max_hidden, max_out)
        self.proj_bn = DynBN(max_out)
        self.stride = stride

    def forward(self, x, cout, kernel, expand, se):
        cin = x.size(1)
        y = F.relu(self.expand_bn(self.expand_conv(x, cin * expand)))
        y = F.relu(self.dw_bn(self.dw_conv(y, kernel, self.stride)))
        if se:
            y = self.se(y)
        y = self.proj_bn(self.proj_conv(y, cout))
        return x + y if self.stride == 1 and cin == cout else y


class SuperNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, G.STEM_CH, 3, 2, 1, bias=False)
        nn.init.kaiming_normal_(self.stem_conv.weight, mode="fan_out")
        self.stem_bn = DynBN(G.STEM_CH)
        stages, max_in = [], G.STEM_CH
        for s in range(G.N_STAGES):
            max_out = G.max_stage_channels(s)
            stages.append(nn.ModuleList([
                DynMBConv(max_in if b == 0 else max_out, max_out, G.STRIDES[s] if b == 0 else 1)
                for b in range(G.MAX_DEPTH)]))
            max_in = max_out
        self.stages = nn.ModuleList(stages)
        self.head_conv = DynPointwise(max_in, G.HEAD_CH)
        self.head_bn = DynBN(G.HEAD_CH)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(G.HEAD_CH, G.NUM_CLASSES)

    def forward(self, x, g):
        d = G.describe(g)
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        for s, st in enumerate(d["stages"]):
            for b, blk in enumerate(st["blocks"]):
                x = self.stages[s][b](x, st["width"], blk["kernel"], blk["expand"], blk["se"])
        x = F.relu(self.head_bn(self.head_conv(x, G.HEAD_CH)))
        return self.classifier(self.dropout(x.mean((2, 3))))

    @torch.no_grad()
    def extract(self, g):
        """Copy the shared weights of genome g into a plain genome.Net (on CPU).
        Its BatchNorm running statistics are NOT valid yet: call evaluate.recalibrate_bn."""
        net = G.Net(g)
        net.stem_conv.weight.copy_(self.stem_conv.weight)
        _copy_bn(net.stem_bn, self.stem_bn)
        for s, stage in enumerate(net.stages):
            for b, blk in enumerate(stage):
                dyn = self.stages[s][b]
                cin, hidden = blk.expand_conv.in_channels, blk.expand_conv.out_channels
                cout, k = blk.proj_conv.out_channels, blk.dw_conv.kernel_size[0]
                blk.expand_conv.weight.copy_(dyn.expand_conv.weight[:hidden, :cin])
                _copy_bn(blk.expand_bn, dyn.expand_bn)
                blk.dw_conv.weight.copy_(dyn.dw_conv.crop(hidden, k))
                _copy_bn(blk.dw_bn, dyn.dw_bn)
                if blk.se is not None:
                    r = blk.se.fc1.out_channels
                    blk.se.fc1.weight.copy_(dyn.se.fc1.weight[:r, :hidden])
                    blk.se.fc1.bias.copy_(dyn.se.fc1.bias[:r])
                    blk.se.fc2.weight.copy_(dyn.se.fc2.weight[:hidden, :r])
                    blk.se.fc2.bias.copy_(dyn.se.fc2.bias[:hidden])
                blk.proj_conv.weight.copy_(dyn.proj_conv.weight[:cout, :hidden])
                _copy_bn(blk.proj_bn, dyn.proj_bn)
        net.head_conv.weight.copy_(self.head_conv.weight[:, :net.head_conv.in_channels])
        _copy_bn(net.head_bn, self.head_bn)
        net.classifier.weight.copy_(self.classifier.weight)
        net.classifier.bias.copy_(self.classifier.bias)
        return net


def _copy_bn(static_bn, dyn_bn):
    c = static_bn.num_features
    static_bn.weight.copy_(dyn_bn.weight[:c])
    static_bn.bias.copy_(dyn_bn.bias[:c])


def load_supernet(path, device):
    net = SuperNet()
    net.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return net.to(device).eval()


def kd_loss(student_logits, teacher_probs):
    return F.kl_div(F.log_softmax(student_logits, -1), teacher_probs, reduction="batchmean")


def train(args):
    seed_everything(args.seed)
    device = get_device()
    data = Data(args.dataset, args.batch_size, args.workers)
    net = SuperNet().to(device)

    decay = [p for n, p in net.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in net.named_parameters() if p.ndim <= 1]   # BN and biases
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0}],
                          lr=args.lr, momentum=0.9, nesterov=True)
    steps = len(data.train_loader)
    total, warm = args.epochs * steps, max(1, args.warmup * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, total - warm))))

    out_dir = RUNS_DIR / "supernet"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path, log_path = out_dir / f"{args.tag}.pt", out_dir / f"{args.tag}_log.csv"
    start = 0
    if ckpt_path.exists():   # resume
        ck = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start = ck["epoch"] + 1
        print(f"resuming from epoch {start}")

    rng = random.Random(args.seed * 1000 + start)
    g_max, g_min = G.extreme_genome("max"), G.extreme_genome("min")
    for epoch in range(start, args.epochs):
        net.train()
        t0, loss_sum = time.time(), 0.0
        for x, y in data.train_loader:
            x, y = prepare(x, device, TRAIN_RES), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = net(x, g_max)                                   # largest subnet: real labels
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            loss.backward()
            loss_sum += loss.item()
            teacher = logits.detach().softmax(-1)
            for g in [g_min] + [G.random_genome(rng) for _ in range(args.n_random)]:
                res = G.RESOLUTIONS[g[0]]
                xr = x if res == TRAIN_RES else F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
                kd_loss(net(xr, g), teacher).backward()              # others: copy the largest
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()

        row = {"epoch": epoch, "loss_max": loss_sum / steps, "minutes": (time.time() - t0) / 60}
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            from evaluate import quick_check     # imported here to avoid a circular import
            row.update(quick_check(net, data, device))
        print(row)
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "args": vars(args)}, ckpt_path)
        new = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "loss_max", "minutes", "acc_max", "acc_mid", "acc_min"],
                               extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the weight-sharing supernet")
    p.add_argument("--dataset", default="imagenette", choices=["imagenette", "cifar10"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=3)
    # 64, not 128: on a 16 GB Mac, batch 128 makes the sandwich step swap to disk (100+ s per step)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)   # linear scaling: 0.1 at batch 128
    p.add_argument("--wd", type=float, default=4e-5)
    p.add_argument("--n-random", type=int, default=2, help="random subnets per batch (sandwich rule)")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="main")
    train(p.parse_args())
