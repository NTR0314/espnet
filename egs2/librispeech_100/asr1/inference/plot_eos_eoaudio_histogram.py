import pickle
from pathlib import Path
import numpy
import argparse
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('at', help='Path of MFA timing txt file')
parser.add_argument('bu', help='utts to skip bcs either bad or missing in mfa')
parser.add_argument('path_audio_lens')
args = parser.parse_args()

# Load mfa timings
timing_dict = {}
mfa_lines = open(Path(args.at)).readlines()
for i, line in enumerate(mfa_lines):
    line = line.strip()
    print(i, line)
    try:
        utt_id, words, timings = line.split(" ")
    except:
        import pdb;pdb.set_trace()
    last_timing = float(timings.strip('"').split(',')[-2]) * 1000
    timing_dict[utt_id] = last_timing

with open(Path(args.bu)) as f:
    bad_utts = [x.strip() for x in f.readlines()]

diffs = []
with open(Path(args.path_audio_lens)) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        utt_id, num_samples = line.split()
        if utt_id in bad_utts:
            print(f"skipped bad utt: {utt_id}")
            continue
        mfa_timing = timing_dict[utt_id]
        audio_len = int(num_samples) / 16
        diffs.append(audio_len - mfa_timing)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15,15))
ax.hist(diffs,bins=100)
ax.grid()
fig.savefig(f"libri_silience_duration_hist.png", bbox_inches="tight")

