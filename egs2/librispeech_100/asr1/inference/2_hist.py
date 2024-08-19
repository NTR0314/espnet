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

diffs = {}
pickle_results = attn_dir.glob('result*.pkl')

for pickle_result in pickle_results:
    alpha = "1"
    alpha_re = re.search('alpha(\d+).pkl', str(pickle_result))
    if alpha_re != None:
        alpha = alpha_re.group(1)

    diffs[alpha] = []
    with open(pickle_result, "rb") as f:
        r = pickle.load(f)
        for uid, td in r.items():
            diffs[alpha].append(td)
    diffs[alpha] = np.array(diffs[alpha])

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
txt = ""
for alpha, timing_diffs in diffs.items():
    ax.hist(timing_diffs, alpha= 0.5, bins=100, range=(-1000, args.right_bound), label=alpha)
    txt = txt + f"mean_{alpha}, std, var: {timing_diffs.mean():.2f}, {timing_diffs.std():.2f}, {timing_diffs.var():.2f}\n"

ax.grid()
plt.figtext(0.5, -0.2, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend(loc='upper right')
fig.savefig(f"{args.filename_plot}", bbox_inches="tight")
