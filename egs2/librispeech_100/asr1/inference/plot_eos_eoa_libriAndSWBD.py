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

libri_mfa_txt_file = "/project/OML/master_theses/ozink/Waseda/espnet/egs2/librispeech_100/asr1/force_alignments/LibriSpeech/test-clean/mfa_test-clean.txt"
audio_lens_path = "/export/data2/ozink/librispeech_100/raw/test_clean/utt2num_samples"
mfa_eval_path_swbd = "/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/force_alignment/mfa_eval2000_aligned"
bad_utts_swbd = "/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/force_alignment/missing_utts_eval2000.txt"
audio_lens_path_swbd = "/export/data2/ozink/raw/eval2000/utt2num_samples"

# Load mfa libri timings
timing_dict = {}
mfa_lines = open(Path(libri_mfa_txt_file)).readlines()
for i, line in enumerate(mfa_lines):
    line = line.strip()
    try:
        utt_id, words, timings = line.split(" ")
    except:
        import pdb;pdb.set_trace()
    last_timing = float(timings.strip('"').split(',')[-2]) * 1000
    timing_dict[utt_id] = last_timing

# Load mfa swbd timings
timing_dict_swbd = {}
mfa_files = os.listdir(mfa_eval_path_swbd)
for file in mfa_files:
    if ".TextGrid" not in file:
        print(f"Skipping file {file}")
        continue
    utt_id = file.split(".", maxsplit=1)[0]
    tg = textgrid.TextGrid.fromFile(Path(mfa_eval_path_swbd) / file)
    spans = [x for x in tg[0] if x.mark != '']
    last_timing = spans[-1].maxTime * 1000 # in ms
    timing_dict_swbd[utt_id] = last_timing

# Calculate difference between audiolen and force alignment. LIBRI
diffs = []
with open(Path(audio_lens_path)) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        utt_id, num_samples = line.split()
        mfa_timing = timing_dict[utt_id]
        audio_len = int(num_samples) / 16
        diffs.append(audio_len - mfa_timing)

with open(Path(bad_utts_swbd)) as f:
    bad_utts = [x.strip() for x in f.readlines()]
diffs_swbd = []
with open(Path(audio_lens_path_swbd)) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        utt_id, num_samples = line.split()
        if utt_id in bad_utts:
            print(f"skipped bad utt: {utt_id}")
            continue
        mfa_timing = timing_dict_swbd[utt_id]
        audio_len = int(num_samples) / 16
        diffs_swbd.append(audio_len - mfa_timing)

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
ax.hist(diffs, alpha=0.5, bins=100, range=(0, 1500), color=colors[0])
ax.set_ylabel('count librispeech', color=colors[0])
ax2 = ax.twinx()
ax2.hist(diffs_swbd, alpha=0.5, bins=100, range=(0, 1500), color=colors[1])
ax2.set_ylabel('count switchboard', color=colors[1])
ax.grid()
fig.savefig(f"eou_eoa_both.png", bbox_inches="tight")

