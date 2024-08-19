import os
import pickle
from pathlib import Path
import numpy as np
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('attn_path', help="attn_path DISTILL")
parser.add_argument('attn_path2', help="attn_path EXPLICIT TIMING LOSS")
parser.add_argument('filename_plot', help="")
args = parser.parse_args()
attn_dir = Path(args.attn_path)
attn_dir2 = Path(args.attn_path2)

diffs = []
diffs_timing = []
with open(attn_dir / 'result.pkl', 'rb') as f:
        r1 = pickle.load(f)
        for uid, td in r1.items():
            diffs.append(td)
with open(attn_dir2 / 'result.pkl', 'rb') as f:
        r1 = pickle.load(f)
        for uid, td in r1.items():
            diffs_timing.append(td)

diffs = np.array(diffs)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(diffs, bins=100, color="skyblue", alpha=0.5, label='Distillation loss: Prediciton differences', range=(-1000, 1000))
ax.hist(diffs_timing, bins=100, color="lightcoral", alpha=0.5, label='Explicit Timing loss: Prediciton differences', range=(-1000, 1000))
ax.grid()
fig.savefig(f"{args.filename_plot}", bbox_inches="tight")
