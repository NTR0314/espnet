import glob
import numpy as np
import pickle
import re
from pathlib import Path
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("base_path", help=" should look sth like ../exp/asr_26_raw_en_bpe5000_sp/")
parser.add_argument("--is_swbd", action='store_true')
args = parser.parse_args()
bp = args.base_path
is_swbd = args.is_swbd
print(f"{is_swbd=}")
test_dir = "eval2000" if is_swbd else "test_clean"

# Normal WER + Timings
base_paths = glob.glob(str(Path(bp) / "*ms_asr_model_valid.acc.ave/"))
results = {}
results_wer = {}
results_wer_masked_only = {}
for base_path in base_paths:
    diffs = []
    mask_duration = re.search(r'(\d+)ms', base_path).group(1)
    attn_path = Path(base_path) / test_dir / "attn_dir" / "result.pkl"
    with open(attn_path, 'rb') as f:
            r1 = pickle.load(f)
            for uid, td in r1.items():
                diffs.append(td)
    diffs = np.array(diffs)
    results[int(mask_duration)] = diffs
    print(diffs.mean(), mask_duration)
    print()

    wer_path = Path(base_path) / test_dir / "score_wer" / "result.txt"
    with open(wer_path, 'rb') as f:
        wer_lines = f.readlines()
    wer = float([x for x in wer_lines if 'Mean' in str(x)][0].split()[-3])
    results_wer[int(mask_duration)] = wer

# masked WER
base_paths = glob.glob(str(Path(bp) / "*ms_onlyMasked_asr_model_valid.acc.ave/"))
for base_path in base_paths:
    wer_path = Path(base_path) / test_dir / "score_wer" / "result.txt"
    mask_duration = re.search(r'(\d+)ms', base_path).group(1)
    with open(wer_path, 'rb') as f:
        wer_lines = f.readlines()
    try:
        wer = float([x for x in wer_lines if 'Mean' in str(x)][0].split()[-3])
    except:
        wer = 0
    results_wer_masked_only[int(mask_duration)] = wer

sorted_items = sorted(results.items())
sorted_wers = sorted(results_wer.items())
sorted_wers_masked_only = sorted(results_wer_masked_only.items())
labels = [item[0] for item in sorted_items]
values = [item[1] for item in sorted_items]
wers = [item[1] for item in sorted_wers]
wers_masked_only = [item[1] for item in sorted_wers_masked_only]
positions = range(1, len(wers) + 1)

color1, color2, color3 = plt.cm.viridis([0, .5, .9])
fig, ax = plt.subplots(1,1, figsize=(30, 15))
p1 = ax.boxplot(values, labels=labels, showfliers=False, positions=positions)
fig.suptitle(f'{bp} test results')
ax.set_ylabel('Absolute timing prediction difference in ms')
ax.set_xlabel('Masking duration from end of utterance in ms')
wer_ax = ax.twinx()
wer_mo_ax = ax.twinx()
wer_mo_ax.spines['right'].set_position(('outward', 60))
# wer_mo_ax.annotate("masked WER is undefined for hyp lengths of 0", xy=(0.4, 0.8))
wer_ax.set_ylabel('WER')
wer_mo_ax.set_ylabel('WER masked tokens only')
p2 = wer_ax.plot(positions, wers, marker='D', label="WER of all tokens.", color=color1)
p3 = wer_mo_ax.plot(positions, wers_masked_only, marker='D', label="WER of masked tokens only. Half masked tokens are included.", color=color2)

wer_ax.yaxis.label.set_color(p2[0].get_color())
wer_mo_ax.yaxis.label.set_color(p3[0].get_color())

fig.legend()
plt.savefig(f"koba_plot_{Path(bp).name}.png", bbox_inches='tight')
