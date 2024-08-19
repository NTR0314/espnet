import pickle
import textgrid
from pathlib import Path
import numpy as np
import argparse
import re
import os
from tqdm import tqdm

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
    timing_dict[utt_id] = [(x.minTime * 1000, x.maxTime * 1000) for x in spans]

with open(Path(args.bu)) as f:
    bad_utts = [x.strip() for x in f.readlines()]

num_masked_words = {}
m_times = [x * 150 for x in list(range(1,11))]
for m_time in m_times:
    num_masked_words[str(m_time)] = []

total_utts = 0
with open(Path(args.path_audio_lens)) as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip()
        utt_id, num_samples = line.split()
        if utt_id in bad_utts:
            print(f"skipped bad utt: {utt_id}")
            continue
        total_utts += 1
        audio_len = int(num_samples) / 16
        for m_time in m_times:
            m_ws_in_utt = 0
            m_start = audio_len - m_time
            m_stop = audio_len
            print(f"{m_start=}", end=' ')
            print(f"{m_stop=}")
            for w_start, w_stop in timing_dict[utt_id]:
                print(f"({w_start}", end=' ')
                print(f"{w_stop}),", end='')
                if w_start >= m_start:
                    m_ws_in_utt += 1.
                elif w_stop > m_start:
                    m_ws_in_utt += (w_stop - m_start) / (w_stop - w_start)
            print(f"{m_ws_in_utt=}", end='\n\n')
            num_masked_words[str(m_time)].append(m_ws_in_utt)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(len(m_times), figsize=(10, 110))
for ax, m_time in zip(axes, m_times):
    mean = np.mean(np.array(num_masked_words[str(m_time)]))
    ax.hist(num_masked_words[str(m_time)],bins=100)
    ax.set_title(f"Masking duration: {m_time}. Mean = {mean}.")
fig.savefig(f"plot_num_masked_tokens.svg", bbox_inches="tight")
