"""
This script is used to generate the plot discussed with Mr. Kobayashi including timing prediction results and WER rates.

"""
import glob
import matplotlib
import numpy as np
import pickle
import re
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-bp", "--base_path", help="default model")
parser.add_argument("-bp2", "--base_path2", help="masked model")
parser.add_argument("--figname")
parser.add_argument("--alpha", help="Only print this alpha")
parser.add_argument("mWER_score_file_bp", help="10best20beamdefault: 6 Zeilen mit mWER fuer 0, 100, ... 500 ms Masking")
parser.add_argument("mWER_score_file_bp_5best20beam", help="5best20beamdefault: 6 Zeilen mit mWER fuer 0, 100, ... 500 ms Masking")
parser.add_argument("mWER_score_file_bp_1best1beam", help="1best1beamdefault: 6 Zeilen mit mWER fuer 0, 100, ... 500 ms Masking")

parser.add_argument("mWER_score_file_bp2", help="10best20beam masked: 6 Zeilen mit mWER fuer 0, 100, ... 500 ms Masking")
parser.add_argument("mWER_score_file_bp2_5best20beam", help="5best20beam masked: 6 Zeilen mit mWER fuer 0, 100, ... 500 ms Masking")
parser.add_argument("mWER_score_file_bp2_1best1beam", help="1best1beam masked: 6 Zeilen mit mWER fuer 0, 100, ... 500 ms Masking")
parser.add_argument("--swbd", action="store_true")
args = parser.parse_args()

if args.swbd:
    print("Using SWBD")

# bp1
mask_paths = glob.glob(args.base_path + "*ms_asr_model_valid.acc.ave/")
results_bp_timing = {}
results_bp_wer_masked_only = {}
results_bp_wer_masked_only_5best20beam = {}
results_bp_wer_masked_only_1best1beam = {}
for mask_path in mask_paths:
    # Timings
    diffs = {}
    if args.swbd:
        attn_dir = Path(mask_path) / "eval2000" / "attn_dir"
    else:
        attn_dir = Path(mask_path) / "test_clean" / "attn_dir"

    pickle_results = attn_dir.glob('result*.pkl')
    mask_duration = re.search(r'(\d+)ms', mask_path).group(1)

    for pickle_result in pickle_results:
        alpha = "1"
        alpha_re = re.search('alpha(\d+).pkl', str(pickle_result))
        if alpha_re != None:
            alpha = alpha_re.group(1)
        if args.alpha != None:
            if alpha != args.alpha:
                continue

        diffs[alpha] = []
        with open(pickle_result, "rb") as f:
            r = pickle.load(f)
            for uid, td in r.items():
                diffs[alpha].append(td)
        diffs[alpha] = np.array(diffs[alpha])
    num_alphas = len(diffs)

    results_bp_timing[int(mask_duration)] = diffs

    # WER
    # wer_path = Path(mask_path) / "test_clean" / "score_wer" / "result.txt"
    # with open(wer_path, 'rb') as f:
    #     wer_lines = f.readlines()
    # wer = float([x for x in wer_lines if 'Mean' in str(x)][0].split()[-3])
    # results_bp_wer[int(mask_duration)] = wer

# Masked WER -> manuell
with open(args.mWER_score_file_bp) as f:
    lines = [x.strip() for x in f.readlines()]
    for mask_duration, line in zip([0, 100, 200, 300, 400, 500], lines):
        mWER = float(line)
        results_bp_wer_masked_only[mask_duration] = mWER
with open(args.mWER_score_file_bp_5best20beam) as f:
    lines = [x.strip() for x in f.readlines()]
    for mask_duration, line in zip([0, 100, 200, 300, 400, 500], lines):
        mWER = float(line)
        results_bp_wer_masked_only_5best20beam[mask_duration] = mWER
with open(args.mWER_score_file_bp_1best1beam) as f:
    lines = [x.strip() for x in f.readlines()]
    for mask_duration, line in zip([0, 100, 200, 300, 400, 500], lines):
        mWER = float(line)
        results_bp_wer_masked_only_1best1beam[mask_duration] = mWER

# bp2
mask_paths = glob.glob(args.base_path2 + "*ms_asr_model_valid.acc.ave/")
results_bp2_timing = {}
results_bp2_wer = {}
results_bp2_wer_masked_only = {}
results_bp2_wer_masked_only_5best20beam = {}
results_bp2_wer_masked_only_1best1beam = {}
for mask_path in mask_paths:
    # Timings
    diffs = {}
    if args.swbd:
        attn_dir = Path(mask_path) / "eval2000" / "attn_dir"
    else:
        attn_dir = Path(mask_path) / "test_clean" / "attn_dir"

    pickle_results = attn_dir.glob('result*.pkl')
    mask_duration = re.search(r'(\d+)ms', mask_path).group(1)

    for pickle_result in pickle_results:
        alpha = "1"
        alpha_re = re.search('alpha(\d+).pkl', str(pickle_result))
        if alpha_re != None:
            alpha = alpha_re.group(1)
        if args.alpha != None:
            if alpha != args.alpha:
                continue

        diffs[alpha] = []
        with open(pickle_result, "rb") as f:
            r = pickle.load(f)
            for uid, td in r.items():
                diffs[alpha].append(td)
        diffs[alpha] = np.array(diffs[alpha])
    num_alphas = len(diffs)

    results_bp2_timing[int(mask_duration)] = diffs

    # WER
    # wer_path = Path(mask_path) / "test_clean" / "score_wer" / "result.txt"
    # with open(wer_path, 'rb') as f:
    #     wer_lines = f.readlines()
    # wer = float([x for x in wer_lines if 'Mean' in str(x)][0].split()[-3])
    # results_bp2_wer[int(mask_duration)] = wer

# Masked WER -> manuell
with open(args.mWER_score_file_bp2) as f:
    lines = [x.strip() for x in f.readlines()]
    for mask_duration, line in zip([0, 100, 200, 300, 400, 500], lines):
        mWER = float(line)
        results_bp2_wer_masked_only[mask_duration] = mWER
with open(args.mWER_score_file_bp2_5best20beam) as f:
    lines = [x.strip() for x in f.readlines()]
    for mask_duration, line in zip([0, 100, 200, 300, 400, 500], lines):
        mWER = float(line)
        results_bp2_wer_masked_only_5best20beam[mask_duration] = mWER
with open(args.mWER_score_file_bp2_1best1beam) as f:
    lines = [x.strip() for x in f.readlines()]
    for mask_duration, line in zip([0, 100, 200, 300, 400, 500], lines):
        mWER = float(line)
        results_bp2_wer_masked_only_1best1beam[mask_duration] = mWER


sorted_bp_timing = sorted(results_bp_timing.items())[:6]
sorted_bp_timing2 = sorted(results_bp2_timing.items())[:6]

# print improvements
diff_diffs = []
for (i1, i2) in zip(sorted_bp_timing, sorted_bp_timing2):
    assert i1[0] == i2[0] #keys?
    # Hoffe timings sind sorted
    diffs = []
    for t1, t2 in zip([x for x in i1[1].values()][0], [x for x in i2[1].values()][0]):
        diff = abs(t1) - abs(t2)
        diffs.append(diff)
    diffs = np.array(diffs)
    print(diffs.mean())
    diff_diffs.append(diffs.mean())
diff_diffs=np.array(diff_diffs)
print(diff_diffs.mean())

diffs_dicts_alphas_bp = [item[1] for item in sorted_bp_timing]
diffs_dicts_alphas_bp2 = [item[1] for item in sorted_bp_timing2]

# sorted_bp_wer = sorted(results_bp_wer.items())
# sorted_bp_wer2 = sorted(results_bp2_wer.items())
# # Only consider until 500ms, ie, first 6 elements including 0ms
# wers = [item[1] for item in sorted_bp_wer][:6]
# wers2 = [item[1] for item in sorted_bp_wer2][:6]

sorted_wers_masked_only = sorted(results_bp_wer_masked_only.items())
sorted_wers_masked_only_5best20beam = sorted(results_bp_wer_masked_only_5best20beam.items())
sorted_wers_masked_only_1best1beam = sorted(results_bp_wer_masked_only_1best1beam.items())

sorted_wers_masked_only2 = sorted(results_bp2_wer_masked_only.items())
sorted_wers_masked_only2_5best20beam = sorted(results_bp2_wer_masked_only_5best20beam.items())
sorted_wers_masked_only2_1best1beam = sorted(results_bp2_wer_masked_only_1best1beam.items())

wers_masked_only = [item[1] for item in sorted_wers_masked_only]
wers_masked_only_5best20beam = [item[1] for item in sorted_wers_masked_only_5best20beam]
wers_masked_only_1best1beam = [item[1] for item in sorted_wers_masked_only_1best1beam]

wers_masked_only2 = [item[1] for item in sorted_wers_masked_only2]
wers_masked_only2_5best20beam = [item[1] for item in sorted_wers_masked_only2_5best20beam]
wers_masked_only2_1best1beam = [item[1] for item in sorted_wers_masked_only2_1best1beam]

labels = [item[0] for item in sorted_bp_timing]
color1, color2, color3 = plt.cm.viridis([0, .5, .9])

fig, ax = plt.subplots(1,1, figsize=(30, 15))

# Define a custom key function
def custom_key(s):
    # Convert the string to a float, assuming the first character is before the decimal point
    if len(s) == 1:
        return float(s)
    else:
        return float(s) / (10 ** len(s))


# In this comparison we only use 1 alpha -> hardcode use color1 and color2 for comparison
positions_bp = range(0, len(wers_masked_only) * 2, 2)
positions_bp2 = [float(x) + 0.1 for x in positions_bp]
positions_bp_neu = [float(x) - 0.1 for x in positions_bp]
# positions_bp2 = range(1, len(wers) * 2, 2)
for i, (masking_time, timing_alpha_dict) in enumerate(sorted_bp_timing):
    for alpha, diff_values in timing_alpha_dict.items():
        bp = ax.boxplot(diff_values, showfliers=False, positions=[positions_bp_neu[i]], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color1)
for i, (masking_time, timing_alpha_dict) in enumerate(sorted_bp_timing2):
    for alpha, diff_values in timing_alpha_dict.items():
        bp = ax.boxplot(diff_values, showfliers=False, positions=[positions_bp2[i]], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color2)


# Font size
font = {'size'   : 22}

matplotlib.rc('font', **font)
plt.rc('font', **font)

fontsize='large'
# ax.set_xlabel('Masking duration from end of utterance in ms')
# wer_ax = ax.twinx()
wer_mo_ax = ax.twinx()
# wer_mo_ax = ax.twinx()
# wer_mo_ax.spines['right'].set_position(('outward', 60))
p1 = wer_mo_ax.plot(positions_bp[1:], wers_masked_only[1:], 'D-', label="mWER default 10best 20beam", color=color1)
p1 = wer_mo_ax.plot(positions_bp[1:], wers_masked_only_5best20beam[1:], 'D--', label="mWER default 5best 20beam", color=color1)
p1 = wer_mo_ax.plot(positions_bp[1:], wers_masked_only_1best1beam[1:], 'D:', label="mWER default 1best 1beam", color=color1)

p1 = wer_mo_ax.plot(positions_bp[1:], wers_masked_only2[1:], 'D-', label="mWER masked 10best 20beam", color=color2)
p1 = wer_mo_ax.plot(positions_bp[1:], wers_masked_only2_5best20beam[1:], 'D--', label="mWER masked 5best 20beam", color=color2)
p1 = wer_mo_ax.plot(positions_bp[1:], wers_masked_only2_1best1beam[1:], 'D:', label="mWER masked 1best 1beam", color=color2)

# wer_mo_ax.annotate("masked WER is undefined for hyp lengths of 0", xy=(0.4, 0.8))
# wer_ax.set_ylabel('WER')
wer_mo_ax.set_ylabel('mWER')
ax.set_xlabel('Masking duration starting from end-of-utterance in ms', fontsize=fontsize)
ax.set_ylabel('Absolute timing prediction difference in ms', fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=22)
ax.tick_params(axis='both', which='minor', labelsize=22)
# p2 = wer_ax.plot(positions_bp, wers, marker='D', label="WER default 1best 1beam", color=color1)
# p2 = wer_ax.plot(positions_bp, wers2, marker='D', label="WER masked 1best 1beam", color=color2)
# wer_ax.yaxis.label.set_color(p2[0].get_color())
ax.axhline(y=0, color='black')
ax.set_xticks(positions_bp, labels)
fig.legend()
plt.savefig(f"{args.figname}.png", bbox_inches='tight')
