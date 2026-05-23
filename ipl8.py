import numpy as np
import cv2
from math import sqrt, cos, pi, ceil
from collections import Counter
import heapq

# =========================================================
# QUANTIZATION MATRICES
# =========================================================

LUM = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)

CHR = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
], dtype=np.float32)


def rgb2ycc(img):

    img = img.astype(np.float32)

    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]

    Y  = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.168736*R - 0.331264*G + 0.5*B + 128
    Cr = 0.5*R - 0.418688*G - 0.081312*B + 128

    return Y, Cb, Cr


def chroma_subsample(channel):

    return channel[::2, ::2]


def zero_center(channel):

    return channel - 128


def dct_2d(block):

    out = np.zeros((8,8), dtype=np.float32)

    for u in range(8):

        for v in range(8):

            cu = 1/sqrt(2) if u == 0 else 1
            cv = 1/sqrt(2) if v == 0 else 1

            s = 0

            for x in range(8):

                for y in range(8):

                    s += (
                        block[x,y]
                        * cos((2*x+1)*u*pi/16)
                        * cos((2*y+1)*v*pi/16)
                    )

            out[u,v] = 0.25 * cu * cv * s

    return out


def zigzag(block):

    indices = sorted(
        ((x,y) for x in range(8) for y in range(8)),
        key=lambda p: (p[0]+p[1], -p[0] if (p[0]+p[1]) % 2 else p[0])
    )

    return [block[x,y] for x,y in indices]


def trim(block):

    block = list(block)

    while len(block) > 0 and block[-1] == 0:
        block.pop()

    return block if block else [0]


def run_length_encode(blocks):

    output = []

    for block in blocks:

        block = trim(block)

        zeros = 0

        for i, val in enumerate(block):

            val = int(val)

            if i == 0:

                output.append(("DC", val))

            else:

                if val == 0:

                    zeros += 1

                else:

                    output.append(("AC", zeros, val))
                    zeros = 0

        output.append(("EOB",))

    return output


class Node:

    def __init__(self, freq, symbol=None, left=None, right=None):

        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):

        return self.freq < other.freq


def build_huffman_tree(data):

    freq = Counter(data)

    heap = []

    for sym, count in freq.items():

        heapq.heappush(heap, Node(count, sym))

    while len(heap) > 1:

        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        merged = Node(left.freq + right.freq, None, left, right)

        heapq.heappush(heap, merged)

    return heap[0]


def generate_codes(node, current="", codes={}):

    if node is None:
        return

    if node.symbol is not None:

        codes[node.symbol] = current

    generate_codes(node.left, current+"0", codes)
    generate_codes(node.right, current+"1", codes)

    return codes


def pad_channel(channel):

    h, w = channel.shape

    new_h = ceil(h/8)*8
    new_w = ceil(w/8)*8

    padded = np.zeros((new_h, new_w), dtype=np.float32)

    padded[:h,:w] = channel

    return padded


def process_channel(channel, qtable):

    padded = pad_channel(channel)

    h, w = padded.shape

    blocks = []

    for i in range(0, h, 8):

        for j in range(0, w, 8):

            block = padded[i:i+8, j:j+8]

            dct_block = dct_2d(block)

            quantized = np.round(dct_block / qtable)

            zz = zigzag(quantized)

            blocks.append(zz)

    return blocks
