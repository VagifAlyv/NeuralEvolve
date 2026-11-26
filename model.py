"""The search space, the genome, and the two networks: a normal network and the supernetwork.

A genome is a list of 31 numbers that describes one network:

    genome[0]               input resolution in pixels
    then for each of the 6 stages, 5 numbers:
        depth               how many blocks the stage has
        width               multiplier for the number of channels
        kernel size         size of the depthwise convolution
        expansion           how much a block widens its input inside
        bits                weight precision after compression (16 = not compressed)

Every number in a genome is always used, so two different genomes are always two different networks.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

RESOLUTIONS = [96, 112, 128]
DEPTHS = [1, 2, 3, 4]
WIDTHS = [0.75, 1.0, 1.25]
KERNEL_SIZES = [3, 5, 7]
EXPANSIONS = [3, 4, 6]
BITS = [4, 6, 8, 16]

STAGE_CHANNELS = [16, 24, 40, 80, 112, 160]  # channels of each stage at width 1.0
STAGE_STRIDES = [1, 2, 2, 2, 1, 2]           # stride 2 halves the image size
STEM_CHANNELS = 16
HEAD_CHANNELS = 640
NUMBER_OF_CLASSES = 10
NUMBER_OF_STAGES = len(STAGE_CHANNELS)

# The options of every position in the genome.
GENE_OPTIONS = [RESOLUTIONS] + [DEPTHS, WIDTHS, KERNEL_SIZES, EXPANSIONS, BITS] * NUMBER_OF_STAGES
SMALLEST_GENOME = [min(options) for options in GENE_OPTIONS]
LARGEST_GENOME = [max(options) for options in GENE_OPTIONS]


def describe(genome):
    """Turn a genome into its resolution and a readable list of stages."""
    resolution = genome[0]
    stages = []
    for stage_number in range(NUMBER_OF_STAGES):
        start = 1 + 5 * stage_number
        depth, width, kernel_size, expansion, bits = genome[start:start + 5]
        channels = int(STAGE_CHANNELS[stage_number] * width / 4 + 0.5) * 4   # round to a multiple of 4
        stages.append({"depth": depth, "channels": channels, "kernel_size": kernel_size,
                       "expansion": expansion, "bits": bits})
    return resolution, stages


def random_genome(generator):
    return [generator.choice(options) for options in GENE_OPTIONS]


def mutate(genome, generator):
    """Each gene gets a new random option with probability 1 / 31, so about one gene changes."""
    child = list(genome)
    for position, options in enumerate(GENE_OPTIONS):
        if generator.random() < 1 / len(GENE_OPTIONS):
            child[position] = generator.choice(options)
    return child


def crossover(mother, father, generator):
    """Uniform crossover: every gene comes from the mother or the father with probability 1/2."""
    child = []
    for mother_gene, father_gene in zip(mother, father):
        if generator.random() < 0.5:
            child.append(mother_gene)
        else:
            child.append(father_gene)
    return child


class Block(nn.Module):
    """MobileNetV2 block: 1x1 convolution to widen, depthwise convolution, 1x1 convolution to narrow."""

    def __init__(self, in_channels, out_channels, kernel_size, expansion, stride):
        super().__init__()
        hidden_channels = in_channels * expansion
        self.stride = stride
        self.expand = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_channels)
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride,
                                   padding=kernel_size // 2, groups=hidden_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_channels)
        self.project = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        outputs = F.relu(self.expand_bn(self.expand(inputs)))
        outputs = F.relu(self.depthwise_bn(self.depthwise(outputs)))
        outputs = self.project_bn(self.project(outputs))
        if self.stride == 1 and inputs.shape[1] == outputs.shape[1]:
            outputs = outputs + inputs   # residual connection
        return outputs


class Network(nn.Module):
    """The normal network described by a genome."""

    block_type = Block

    def __init__(self, genome):
        super().__init__()
        resolution, stages = describe(genome)
        self.stem = nn.Conv2d(3, STEM_CHANNELS, 3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(STEM_CHANNELS)
        self.stages = nn.ModuleList()
        in_channels = STEM_CHANNELS
        for stage_number, stage in enumerate(stages):
            blocks = nn.ModuleList()
            for block_number in range(stage["depth"]):
                stride = STAGE_STRIDES[stage_number] if block_number == 0 else 1
                blocks.append(self.block_type(in_channels, stage["channels"], stage["kernel_size"],
                                              stage["expansion"], stride))
                in_channels = stage["channels"]
            self.stages.append(blocks)
        self.head = nn.Conv2d(in_channels, HEAD_CHANNELS, 1, bias=False)
        self.head_bn = nn.BatchNorm2d(HEAD_CHANNELS)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(HEAD_CHANNELS, NUMBER_OF_CLASSES)

    def forward(self, images):
        features = F.relu(self.stem_bn(self.stem(images)))
        for blocks in self.stages:
            for block in blocks:
                features = block(features)
        features = F.relu(self.head_bn(self.head(features)))
        features = features.mean(dim=(2, 3))
        return self.classifier(self.dropout(features))


# ------------------------------------------------------------------ supernetwork
#
# The supernetwork is the largest network. Every smaller network uses a part of its weights:
#   fewer blocks       -> the first blocks of each stage
#   fewer channels     -> the first channels of each layer
#   smaller kernel     -> the center of the 7x7 kernel
# So one training run trains all networks at once.

def shared_batchnorm(features, batchnorm):
    """BatchNorm with the first channels of the parameters, always using the statistics of the batch.
    A network taken out of the supernetwork gets its own statistics later (see search.py)."""
    channels = features.shape[1]
    return F.batch_norm(features, None, None, batchnorm.weight[:channels], batchnorm.bias[:channels],
                        training=True)


class SuperBlock(Block):
    """A block of the largest size that can run as any smaller block."""

    def forward(self, inputs, out_channels, kernel_size, expansion):
        in_channels = inputs.shape[1]
        hidden_channels = in_channels * expansion
        largest_kernel_size = self.depthwise.weight.shape[2]
        start = (largest_kernel_size - kernel_size) // 2
        end = start + kernel_size

        outputs = F.conv2d(inputs, self.expand.weight[:hidden_channels, :in_channels])
        outputs = F.relu(shared_batchnorm(outputs, self.expand_bn))
        depthwise_weight = self.depthwise.weight[:hidden_channels, :, start:end, start:end]
        outputs = F.conv2d(outputs, depthwise_weight, stride=self.stride, padding=kernel_size // 2,
                           groups=hidden_channels)
        outputs = F.relu(shared_batchnorm(outputs, self.depthwise_bn))
        outputs = F.conv2d(outputs, self.project.weight[:out_channels, :hidden_channels])
        outputs = shared_batchnorm(outputs, self.project_bn)
        if self.stride == 1 and in_channels == out_channels:
            outputs = outputs + inputs
        return outputs


class SuperNetwork(Network):
    block_type = SuperBlock

    def __init__(self):
        super().__init__(LARGEST_GENOME)

    def forward(self, images, genome):
        resolution, stages = describe(genome)
        features = F.relu(shared_batchnorm(self.stem(images), self.stem_bn))
        for stage_number, stage in enumerate(stages):
            for block_number in range(stage["depth"]):
                block = self.stages[stage_number][block_number]
                features = block(features, stage["channels"], stage["kernel_size"], stage["expansion"])
        in_channels = features.shape[1]
        features = F.conv2d(features, self.head.weight[:, :in_channels])
        features = F.relu(shared_batchnorm(features, self.head_bn))
        features = features.mean(dim=(2, 3))
        return self.classifier(self.dropout(features))


@torch.no_grad()
def extract(supernetwork, genome):
    """Copy the weights a genome uses out of the supernetwork into a normal Network.
    The layers have the same names in both networks, so we only cut every weight to the smaller shape."""
    network = Network(genome)
    large_weights = supernetwork.state_dict()
    for name, small_weight in network.state_dict().items():
        large_weight = large_weights[name]
        if small_weight.dim() == 4:   # convolution: [out channels, in channels, kernel, kernel]
            out_channels, in_channels, kernel_size, _ = small_weight.shape
            start = (large_weight.shape[2] - kernel_size) // 2
            end = start + kernel_size
            small_weight.copy_(large_weight[:out_channels, :in_channels, start:end, start:end])
        elif small_weight.dim() == 2:   # the classifier has the same shape in every network
            small_weight.copy_(large_weight)
        elif small_weight.dim() == 1:   # BatchNorm and bias vectors
            small_weight.copy_(large_weight[:small_weight.shape[0]])
    return network


if __name__ == "__main__":
    generator = random.Random(0)
    supernetwork = SuperNetwork()
    supernetwork.dropout.eval()   # turn off dropout so both networks give the same output
    for genome in [SMALLEST_GENOME, LARGEST_GENOME, random_genome(generator)]:
        resolution = genome[0]
        network = extract(supernetwork, genome)
        images = torch.randn(4, 3, resolution, resolution)
        network.train()   # BatchNorm uses the batch statistics, like in the supernetwork
        network.dropout.eval()
        difference = (network(images) - supernetwork(images, genome)).abs().max().item()
        parameters = sum(parameter.numel() for parameter in network.parameters())
        print(f"resolution {resolution}  parameters {parameters / 1e6:.2f} M  "
              f"difference to supernetwork {difference:.1e}")
