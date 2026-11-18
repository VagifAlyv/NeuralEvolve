import random
import torch
import torch.nn as nn

RESOLUTIONS = [96,112,128] # picture size
DEPTHS = [1,2,3,4] # how many blocks
WIDTH_MULTS = [0.75, 1, 1.25] 
BITS = [4,6,8,16]

KERNELS = [3,5,7]
EXPANDS = [3,4,6]
SE = [0,1]

N_STAGES = 6
MAX_DEPTH = 4
NUM_CLASSES = 10
STAGE_OFF = 1
BLOCK_OFF = STAGE_OFF + N_STAGES * 3
GENOME_LEN = BLOCK_OFF + N_STAGES * MAX_DEPTH * 3

assert GENOME_LEN == 91
BASE_CH = [16, 24, 40, 80, 112, 160]
STRIDES = [1, 2, 2, 2, 1, 2]

N_CHOICES = ([len(RESOLUTIONS)]
             + [len(DEPTHS), len(WIDTH_MULTS), len(BITS)] * N_STAGES
             + [len(KERNELS), len(EXPANDS), len(SE)] * (N_STAGES * MAX_DEPTH))

assert len(N_CHOICES) == GENOME_LEN

def stage_index(s):
    return STAGE_OFF + 3 * s

def block_index(s):
    return BLOCK_OFF + 3 * (s * MAX_DEPTH + b)


def random_genome(rng = random):
    return [rng.randrange(n) for n in N_CHOICES]