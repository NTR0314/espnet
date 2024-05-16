import os
import pickle
from pathlib import Path
import numpy as np
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('attn_path1', help="")
parser.add_argument('filename_plot', help="")
args = parser.parse_args()
attn_dir1 = Path(args.attn_path1)

diffs = []
real_times = {}

with open(attn_dir1 / 'result.pkl', 'rb') as f:
    with open('/project/OML/master_theses/ozink/Waseda/espnet/egs2/librispeech_100/asr1/dump/raw/test_clean/utt2num_samples') as g:
        r1 = pickle.load(f)
        realz = g.readlines()
        for line in realz:
            a = line.split(' ')
            real_times[a[0]] = int(a[1]) / 16

        for uid, t1 in r1.items():
            t2 = real_times[uid]
            if abs(t2 - t1) >= 30000:
                import pdb; pdb.set_trace()
            diffs.append(t2 - t1)


# plot distributioon
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(diffs, bins=200, range=(-1000, 1000))
ax.grid()
fig.savefig(f"{args.filename_plot}.png")
