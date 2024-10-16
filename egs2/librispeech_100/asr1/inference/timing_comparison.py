"""
Legacy script to compare the timing predictions of two models. (In this case masked/unmasked)
"""

import os
from pathlib import Path
import numpy as np
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('attn_path1', help="")
parser.add_argument('attn_path2', help=" should be the unmasked model in the comparison")
parser.add_argument('out_file_name', help="output_file name")
parser.add_argument('boundaries', default=-1, type=int, help="-1 = default = no boundaries")
args = parser.parse_args()
attn_dir1 = Path(args.attn_path1)
attn_dir2 = Path(args.attn_path2)
file_name = args.out_file_name
bounds=args.boundaries

diffs = []
diffs_non_abs = []
import pickle

with open(attn_dir1 / 'result.pkl', 'rb') as f:
    with open(attn_dir2 / 'result.pkl', 'rb') as g:
        r1 = pickle.load(f)
        r2 = pickle.load(g)

        for uid, t1 in r1.items():
            t2 = r2[uid]
            diffs.append(abs(t2 - t1))
            diffs_non_abs.append(t2 - t1)

diffs = np.array(diffs)
print(f"mean = {np.mean(diffs)}")
print(f"avg = {np.average(diffs)}")
print(f"median = {np.median(diffs)}")
print(f"{np.max(diffs)}")
print(f"{np.min(diffs)}")

# plot distributioon
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
if bounds != -1:
    ax.hist(diffs_non_abs, bins=200, range=(-bounds, bounds))
else:
    ax.hist(diffs_non_abs, bins=200)
plt.xlabel("Timing offset in ms. Positive: masked model predicts less than full context model.")
plt.ylabel("Histogram count")
ax.grid()
fig.savefig(f"{file_name}")
