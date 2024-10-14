"""
This file uses the dumped attention weights during inference to calculate the timing prediction differences using
the MFA timings. It saves an intermediary file called `result.pkl`

This intermediary file is used by other scripts, such as plotting.
"""


import pickle
import textgrid
from pathlib import Path
import numpy
import argparse
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('attn_path', help="Path where the attention dumps are saved, .../attn_dir/")
parser.add_argument('at', help='Path of MFA timing folder')
parser.add_argument('--use_last_head', action='store_true', help="flag if should use last head for argmax, else will use average")
parser.add_argument('--alpha', default=1.0, type=float)
args = parser.parse_args()
attn_dir = Path(args.attn_path)
results = {}

print(f"{args.use_last_head=}")
print(f"{args.alpha=}")

# Load mfa timings
mfa_eval_path = args.at
mfa_files = os.listdir(mfa_eval_path)
timing_dict = {}

# load mfa files
for file in mfa_files:
    if ".TextGrid" not in file:
        print(f"Skipping file {file}")
        continue
    utt_id = file.split(".", maxsplit=1)[0]
    tg = textgrid.TextGrid.fromFile(Path(mfa_eval_path) / file)
    spans = [x for x in tg[0] if x.mark != '']
    last_timing = spans[-1].maxTime * 1000 # in ms
    timing_dict[utt_id] = last_timing

for att_file_name in os.listdir(attn_dir):
    if not ".npy" in att_file_name:
        continue

    # LIBRI: utt_908-31957-0020_step_019.npy
    # SWBD: utt_sw02001-A_000098-001156_step_036.npy
    # SWBD test: utt_en_4966-A_065993-066715_step_040.npy
    # SWBD test: utt_sw_4824-A_004423-004562_step_002.npy
    # SWBD dev: utt_sw02022-B_032570-032898_step_007.npy
    try:
        match = re.search('utt_(e?n?s?w?_?[sw\d]+-[AB_\d]+-\d+)_step_\d+.npy', att_file_name)
        utt_id = match.group(1)
    except:
        print("An unexpected error occured.")
        print(att_file_name)
        import pdb;pdb.set_trace()

    att_w = numpy.load(attn_dir / att_file_name)
    # remove d = 1 dim
    att_w = att_w.squeeze()

    # Take avg/last head for inference
    if not args.use_last_head:
        att_w = numpy.average(att_w, axis=0)
    else:
        att_w = att_w[-1]

    try:
        timing_index = numpy.where(att_w >= att_w.max() * args.alpha)[0][-1]
    except:
        import pdb;pdb.set_trace()
    pred_timing = timing_index * 40 # in ms
    actual_mfa_timing = timing_dict[utt_id]
    results[utt_id] = actual_mfa_timing - pred_timing

# Preview
print(f"{attn_dir}:\t\tavg is: {numpy.array([x for x in iter(results.values())]).mean():.1f}\n")

alpha_str = str(args.alpha).replace('.','')
save_file_name = 'result.pkl' if args.alpha == 1.0 else f'result_alpha{alpha_str}.pkl'
with open(attn_dir / save_file_name, 'wb') as f:
    pickle.dump(results, f)
