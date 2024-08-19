import glob
import numpy as np
import pickle
import re
from pathlib import Path
import matplotlib.pyplot as plt

base_paths = glob.glob("../exp/asr_26_raw_en_bpe5000_sp/*ms_asr_model_valid.acc.ave/")
results = {}
results_wer = {}
results_wer_masked_only = {}
for base_path in base_paths:
    diffs = {}
    attn_dir = Path(base_path) / "test_clean" / "attn_dir"
    pickle_results = attn_dir.glob('result*.pkl')
    mask_duration = re.search(r'(\d+)ms', base_path).group(1)

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
    num_alphas = len(diffs)

    results[int(mask_duration)] = diffs

    # WER
    wer_path = Path(base_path) / "test_clean" / "score_wer" / "result.txt"
    with open(wer_path, 'rb') as f:
        wer_lines = f.readlines()
    wer = float([x for x in wer_lines if 'Mean' in str(x)][0].split()[-3])
    results_wer[int(mask_duration)] = wer

# Masked WER
base_paths = glob.glob("../exp/asr_26_raw_en_bpe5000_sp/*ms_onlyMasked_asr_model_valid.acc.ave/")
for base_path in base_paths:
    wer_path = Path(base_path) / "test_clean" / "score_wer" / "result.txt"
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
diffs_dicts_alphas = [item[1] for item in sorted_items]
wers = [item[1] for item in sorted_wers]
wers_masked_only = [item[1] for item in sorted_wers_masked_only]
positions = range(1 + num_alphas // 2, (len(wers)) * num_alphas, num_alphas)

color1, color2, color3 = plt.cm.viridis([0, .5, .9])
fig, ax = plt.subplots(1,1, figsize=(30, 15))

# Define a custom key function
def custom_key(s):
    # Convert the string to a float, assuming the first character is before the decimal point
    if len(s) == 1:
        return float(s)
    else:
        return float(s) / (10 ** len(s))

color_map = plt.get_cmap('viridis')
colors = [color_map(i / num_alphas) for i in range(num_alphas)]
for i, alphas_dict in enumerate(diffs_dicts_alphas): #[0ms, 100ms ..]
    positions_alpha = range(1 + i * num_alphas, 1 + (i + 1) * num_alphas)
    sorted_dict = {k: alphas_dict[k] for k in sorted(alphas_dict.keys(), key=custom_key, reverse=True)}
    for j, (alpha, diff_values) in enumerate(sorted_dict.items()): #[1, 0.01, 0.05 ..]
        bp = ax.boxplot(diff_values, showfliers=False, positions=[positions_alpha[j]], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(colors[j])
        print(j, f"{alpha=}")

fig.suptitle('Librispeech test results')
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

ax.axhline(y=0)

fig.legend()
plt.savefig("koba_plot_libri.png", bbox_inches='tight')
