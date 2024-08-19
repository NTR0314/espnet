import pickle
import textgrid
from pathlib import Path
import numpy
import argparse
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('at', help='Path of MFA timing folder')
parser.add_argument('bu', help='bad utt txt file path')
parser.add_argument('path_audio_lens')
args = parser.parse_args()

# Load mfa timings
mfa_eval_path = args.at
mfa_files = os.listdir(mfa_eval_path)
timing_dict = {}
for file in mfa_files:
    if ".TextGrid" not in file:
        print(f"Skipping file {file}")
        continue
    utt_id = file.split(".", maxsplit=1)[0]
    tg = textgrid.TextGrid.fromFile(Path(mfa_eval_path) / file)
    spans = [x for x in tg[0] if x.mark != '']
    last_timing = spans[-1].maxTime * 1000 # in ms
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
fig, ax = plt.subplots()
ax.hist(diffs,bins=100)
ax.grid()
fig.savefig(f"plot_eos_eoaudio_histogram.png", bbox_inches="tight")

