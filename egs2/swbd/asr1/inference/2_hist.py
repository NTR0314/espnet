import os
import pickle
from pathlib import Path
import numpy as np
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('attn_path', help="")
parser.add_argument('filename_plot', help="")
parser.add_argument('--right_bound', help="", default=1000, required=False, type=int)
args = parser.parse_args()
attn_dir = Path(args.attn_path)

diffs = []
with open(attn_dir / 'result.pkl', 'rb') as f:
        r1 = pickle.load(f)
        for uid, td in r1.items():
            diffs.append(td)

diffs = np.array(diffs)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(diffs,bins=100, range=(-1000, args.right_bound))
ax.grid()
txt = f"mean, std, var: {diffs.mean():.2f}, {diffs.std():.2f}, {diffs.var():.2f}"
plt.figtext(0.5, -0.1, txt, wrap=True, horizontalalignment='center', fontsize=12)
fig.savefig(f"{args.filename_plot}", bbox_inches="tight")
