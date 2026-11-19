"""Search space, genome and genetic operators, and the plain (static) network a genome describes.

Genome = list of 91 integers. Every integer is an INDEX into one of the option lists below.

    position 0          resolution
    positions 1..18     6 stages x (depth, width multiplier, bits)
    positions 19..90    6 stages x 4 block slots x (kernel, expansion, SE)

Only the architecture and the compression bits are evolved. Weights are always trained by SGD.
"""
import functools
import random

import torch
import torch.nn as nn

RESOLUTIONS = [96, 112, 128]   # input picture size in pixels
DEPTHS = [1, 2, 3, 4]          # how many blocks a stage has
WIDTH_MULTS = [0.75, 1, 1.25]  # scales the number of channels of a stage
BITS = [4, 6, 8, 16]           # weight precision of a stage (16 = no palettization)

KERNELS = [3, 5, 7]            # depthwise kernel size of a block
EXPANDS = [3, 4, 6]            # expansion ratio inside a block
SE = [0, 1]                    # squeeze-excite off / on

N_STAGES = 6
MAX_DEPTH = 4
NUM_CLASSES = 10
STAGE_OFF = 1
BLOCK_OFF = STAGE_OFF + N_STAGES * 3
GENOME_LEN = BLOCK_OFF + N_STAGES * MAX_DEPTH * 3

assert GENOME_LEN == 91
BASE_CH = [16, 24, 40, 80, 112, 160]   # channels of each stage at width multiplier 1.0
STRIDES = [1, 2, 2, 2, 1, 2]           # stride of the first block of each stage (2 = halve the image)
STEM_CH = 16                           # channels after the first conv (fixed)
HEAD_CH = 640                          # channels of the last 1x1 conv before the classifier (fixed)
STEM_HEAD_BITS = 16                    # stem, head and classifier are never palettized
PALETTIZE_MIN_WEIGHTS = 2048           # like Core ML: smaller weight tensors stay at 16 bits

N_CHOICES = ([len(RESOLUTIONS)]
             + [len(DEPTHS), len(WIDTH_MULTS), len(BITS)] * N_STAGES
             + [len(KERNELS), len(EXPANDS), len(SE)] * (N_STAGES * MAX_DEPTH))

assert len(N_CHOICES) == GENOME_LEN


def stage_index(s):
    """Position of the first gene (depth) of stage s. +1 = width, +2 = bits."""
    return STAGE_OFF + 3 * s


def block_index(s, b):
    """Position of the first gene (kernel) of block slot b in stage s. +1 = expansion, +2 = SE."""
    return BLOCK_OFF + 3 * (s * MAX_DEPTH + b)


def make_divisible(v, divisor=4):
    """Round a channel count to the nearest multiple of `divisor` (hardware likes round numbers)."""
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


def stage_channels(s, width_idx):
    return make_divisible(BASE_CH[s] * WIDTH_MULTS[width_idx])


def max_stage_channels(s):
    return stage_channels(s, len(WIDTH_MULTS) - 1)


def se_channels(c):
    """Bottleneck size of a squeeze-excite module on c channels."""
    return make_divisible(c // 4, 8)


# ---------------------------------------------------------------- reading a genome

def describe(g):
    """Turn the 91 indices into real values, e.g. {'resolution': 112, 'stages': [...]}."""
    stages = []
    for s in range(N_STAGES):
        i = stage_index(s)
        depth = DEPTHS[g[i]]
        blocks = []
        for b in range(depth):
            j = block_index(s, b)
            blocks.append({"kernel": KERNELS[g[j]], "expand": EXPANDS[g[j + 1]], "se": SE[g[j + 2]]})
        stages.append({"depth": depth, "width": stage_channels(s, g[i + 1]),
                       "bits": BITS[g[i + 2]], "blocks": blocks})
    return {"resolution": RESOLUTIONS[g[0]], "stages": stages}


def bits_per_stage(g):
    return [BITS[g[stage_index(s) + 2]] for s in range(N_STAGES)]


def active_mask(g):
    """True for genes that change the network. Block genes beyond a stage's depth are inactive."""
    mask = [True] * GENOME_LEN
    for s in range(N_STAGES):
        depth = DEPTHS[g[stage_index(s)]]
        for b in range(depth, MAX_DEPTH):
            j = block_index(s, b)
            mask[j] = mask[j + 1] = mask[j + 2] = False
    return mask


# ---------------------------------------------------------------- genetic operators

def random_genome(rng=random):
    return [rng.randrange(n) for n in N_CHOICES]


def repair(g):
    """Canonical form: set inactive genes to 0, so identical networks get identical genomes.

    Used as a cache key. Inside the GA it is optional (RQ4 'repair on/off'): without it, inactive
    genes are kept and can become active again later (a kind of hidden memory).
    """
    g = [int(x) for x in g]      # pymoo hands us numpy arrays
    return [x if active else 0 for x, active in zip(g, active_mask(g))]


def genome_key(g):
    """String key of the canonical genome, used by caches and logs."""
    return "-".join(map(str, repair(g)))


def mutate(g, rate=None, rng=random):
    """Each gene is reset to a random option with probability `rate` (default 1/91)."""
    rate = 1.0 / GENOME_LEN if rate is None else rate
    out = list(g)
    for i, n in enumerate(N_CHOICES):
        if rng.random() < rate:
            out[i] = rng.randrange(n)
    return out


def crossover_uniform(a, b, rng=random):
    """Each gene comes from parent a or b with probability 1/2."""
    return [x if rng.random() < 0.5 else y for x, y in zip(a, b)]


def crossover_one_point(a, b, rng=random):
    """Genes before a random cut come from a, the rest from b."""
    cut = rng.randrange(1, GENOME_LEN)
    return list(a[:cut]) + list(b[cut:])


def crossover_stagewise(a, b, rng=random):
    """Whole stages (stage genes + its 4 block slots) are inherited together from one parent."""
    child = list(a)
    if rng.random() < 0.5:
        child[0] = b[0]
    for s in range(N_STAGES):
        if rng.random() < 0.5:
            i = stage_index(s)
            child[i:i + 3] = b[i:i + 3]
            j0, j1 = block_index(s, 0), block_index(s, MAX_DEPTH - 1) + 3
            child[j0:j1] = b[j0:j1]
    return child


def extreme_genome(which):
    """'min' = every gene at its smallest option, 'max' = largest, 'mid' = middle option."""
    if which == "min":
        return [0] * GENOME_LEN
    if which == "max":
        return [n - 1 for n in N_CHOICES]
    return [n // 2 for n in N_CHOICES]


# ---------------------------------------------------------------- the network a genome describes

class SqueezeExcite(nn.Module):
    """Learns one weight per channel from the whole image, then rescales the channels."""

    def __init__(self, c):
        super().__init__()
        r = se_channels(c)
        self.fc1 = nn.Conv2d(c, r, 1)
        self.fc2 = nn.Conv2d(r, c, 1)
        self.act = nn.ReLU()
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        return x * self.gate(self.fc2(self.act(self.fc1(s))))


class MBConv(nn.Module):
    """MobileNetV2 block: 1x1 expand -> kxk depthwise -> (SE) -> 1x1 project, plus residual."""

    def __init__(self, cin, cout, kernel, expand, se, stride):
        super().__init__()
        hidden = cin * expand
        self.expand_conv = nn.Conv2d(cin, hidden, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden)
        self.dw_conv = nn.Conv2d(hidden, hidden, kernel, stride, kernel // 2, groups=hidden, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden)
        self.se = SqueezeExcite(hidden) if se else None
        self.proj_conv = nn.Conv2d(hidden, cout, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()
        self.use_residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.act(self.expand_bn(self.expand_conv(x)))
        y = self.act(self.dw_bn(self.dw_conv(y)))
        if self.se is not None:
            y = self.se(y)
        y = self.proj_bn(self.proj_conv(y))
        return x + y if self.use_residual else y


class Net(nn.Module):
    def __init__(self, g, num_classes=NUM_CLASSES):
        super().__init__()
        d = describe(g)
        self.stem_conv = nn.Conv2d(3, STEM_CH, 3, 2, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(STEM_CH)
        cin = STEM_CH
        stages = []
        for s, st in enumerate(d["stages"]):
            blocks = []
            for b, blk in enumerate(st["blocks"]):
                stride = STRIDES[s] if b == 0 else 1
                blocks.append(MBConv(cin, st["width"], blk["kernel"], blk["expand"], blk["se"], stride))
                cin = st["width"]
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.head_conv = nn.Conv2d(cin, HEAD_CH, 1, bias=False)
        self.head_bn = nn.BatchNorm2d(HEAD_CH)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(HEAD_CH, num_classes)

    def forward(self, x):
        x = self.act(self.stem_bn(self.stem_conv(x)))
        for stage in self.stages:
            x = stage(x)
        x = self.act(self.head_bn(self.head_conv(x)))
        x = x.mean((2, 3))
        return self.classifier(self.dropout(x))


def decode(g, num_classes=NUM_CLASSES):
    """Genome -> (untrained PyTorch model, input resolution, bits per stage)."""
    return Net(g, num_classes), RESOLUTIONS[g[0]], bits_per_stage(g)


# ---------------------------------------------------------------- cheap cost measures

def _weight_params(module):
    """Parameters that end up as weights in Core ML (BatchNorm gets folded into the convs)."""
    return sum(p.numel() for m in module.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
               for p in m.parameters(recurse=False))


def count_params(model):
    return _weight_params(model)


def model_size_mb(model, bits):
    """Size after palettization: big stage weights at their stage's bits, everything else at 16 bits."""
    total_bits = _weight_params(model) * STEM_HEAD_BITS
    for stage, b in zip(model.stages, bits):
        for m in stage.modules():
            if isinstance(m, nn.Conv2d) and m.weight.numel() >= PALETTIZE_MIN_WEIGHTS:
                total_bits -= m.weight.numel() * (STEM_HEAD_BITS - b)
    return total_bits / 8 / 1e6


def count_macs(model, resolution):
    """Multiply-accumulate operations for one image (FLOPs ~= 2 x MACs)."""
    macs = 0

    def conv_hook(m, inp, out):
        nonlocal macs
        macs += out.numel() // out.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m, inp, out):
        nonlocal macs
        macs += m.in_features * m.out_features

    hooks = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hooks += [m.register_forward_hook(linear_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, resolution, resolution))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs


@functools.lru_cache(maxsize=200_000)
def _costs_from_key(key):
    g = [int(x) for x in key.split("-")]
    model, res, bits = decode(g)
    return {"params": count_params(model), "macs": count_macs(model, res), "size_mb": model_size_mb(model, bits)}


def cost_measures(g):
    """{'params', 'macs', 'size_mb'} of a genome, cached because the GA asks for the same ones often."""
    return dict(_costs_from_key(genome_key(g)))


if __name__ == "__main__":
    rng = random.Random(0)
    for name, g in [("min", extreme_genome("min")), ("mid", extreme_genome("mid")),
                    ("max", extreme_genome("max")), ("random", random_genome(rng))]:
        model, res, bits = decode(g)
        out = model(torch.randn(2, 3, res, res))
        print(f"{name:6s} res={res} depths={[s['depth'] for s in describe(g)['stages']]} bits={bits} "
              f"params={count_params(model) / 1e6:.2f}M size={model_size_mb(model, bits):.2f}MB "
              f"MMACs={count_macs(model, res) / 1e6:.0f} out={tuple(out.shape)}")
