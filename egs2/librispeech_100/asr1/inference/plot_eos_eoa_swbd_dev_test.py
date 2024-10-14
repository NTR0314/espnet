"""
This script plots the difference of the last MFA timing (EOU) and the length of the audio (based on utt2num_samples) for both dev/test set for SWBD.
"""

import matplotlib.pyplot as plt
import matplotlib
import textgrid
import pickle
from pathlib import Path
import numpy
import argparse
import re
import os

parser = argparse.ArgumentParser()
args = parser.parse_args()

# SWBD Test set
swbd_test_mfa = "/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/force_alignment/mfa_eval2000_aligned"
swbd_test_audio_lens = "/export/data2/ozink/raw/eval2000/utt2num_samples"

# SWBD Dev set
swbd_dev_mfa = "/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/force_alignment/mfa_dev_aligned"
swbd_dev_audio_lens = "/export/data2/ozink/raw/train_dev/utt2num_samples"
swbd_dev_bad_utts = "/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/force_alignment/missing_uids_mfa_dev_swbd.txt"

# Load SWBD Test MFA timings
timing_dict_swbd_test = {}
mfa_files = os.listdir(swbd_test_mfa)
for file in mfa_files:
    if ".TextGrid" not in file:
        print(f"Skipping file {file}")
        continue
    utt_id = file.split(".", maxsplit=1)[0]
    tg = textgrid.TextGrid.fromFile(Path(swbd_test_mfa) / file)
    spans = [x for x in tg[0] if x.mark != '']
    last_timing = spans[-1].maxTime * 1000 # in ms
    timing_dict_swbd_test[utt_id] = last_timing

# Load mfa swbd timings
timing_dict_swbd = {}
mfa_files = os.listdir(swbd_dev_mfa)
for file in mfa_files:
    if ".TextGrid" not in file:
        print(f"Skipping file {file}")
        continue
    utt_id = file.split(".", maxsplit=1)[0]
    tg = textgrid.TextGrid.fromFile(Path(swbd_dev_mfa) / file)
    spans = [x for x in tg[0] if x.mark != '']
    last_timing = spans[-1].maxTime * 1000 # in ms
    timing_dict_swbd[utt_id] = last_timing

# Calculate difference between audiolen and force alignment. test set
diffs_test = []
with open(Path(swbd_test_audio_lens)) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        utt_id, num_samples = line.split()
        mfa_timing = timing_dict_swbd_test[utt_id]
        audio_len = int(num_samples) / 16
        diffs_test.append(audio_len - mfa_timing)

with open(Path(swbd_dev_bad_utts)) as f:
    bad_utts_dev = [x.strip() for x in f.readlines()]
diffs_swbd_dev = []
with open(Path(swbd_dev_audio_lens)) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        utt_id, num_samples = line.split()
        if utt_id in bad_utts_dev:
            print(f"skipped bad utt: {utt_id}")
            continue
        mfa_timing = timing_dict_swbd[utt_id]
        audio_len = int(num_samples) / 16
        diffs_swbd_dev.append(audio_len - mfa_timing)

num_plots = 2
color_map = plt.get_cmap('viridis')
colors = [color_map(i / num_plots) for i in range(num_plots)]

# Font size
font = {
    # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(15,15))
ax.hist(diffs_test, alpha=0.5, bins=100, range=(0, 1500), color=colors[0])
ax2 = ax.twinx()
ax2.hist(diffs_swbd_dev, alpha=0.5, bins=100, range=(0, 1500), color=colors[1])
ax.grid()
fig.savefig(f"eou_eoa_swbd_dev_test.png", bbox_inches="tight")

