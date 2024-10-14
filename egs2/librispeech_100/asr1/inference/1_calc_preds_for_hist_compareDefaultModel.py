"""
This script is similar to `1_calc_preds_for_hist.py` but instead of comparing to the GT MFA timings, we compare to a default model
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
parser.add_argument('attn_path2', help="Path where the attention dumps are saved, .../attn_dir/")
parser.add_argument('--use_last_head', action='store_true', help="flag if should use last head for argmax, else will use average")
parser.add_argument('--alpha', default=1.0, type=float)
args = parser.parse_args()
attn_dir = Path(args.attn_path)
attn_dir2 = Path(args.attn_path2)
results = {}

print(f"{args.use_last_head=}")
print(f"{args.alpha=}")

for att_file_name in os.listdir(attn_dir):
    if not ".npy" in att_file_name:
        continue

    # utt_908-31957-0020_step_019.npy
    match = re.search('utt_(\d+-\d+-\d+)_step_\d+.npy', att_file_name)
    utt_id = match.group(1)
    att_w = numpy.load(attn_dir / att_file_name)
    att_w2 = numpy.load(attn_dir2 / att_file_name)

    # remove d = 1 dim
    att_w = att_w.squeeze()
    att_w2 = att_w2.squeeze()

    # Take avg/last head for inference
    if not args.use_last_head:
        att_w = numpy.average(att_w, axis=0)
        att_w2 = numpy.average(att_w2, axis=0)
    else:
        att_w = att_w[-1]
        att_w2 = att_w2[-1]

    try:
        timing_index = numpy.where(att_w >= att_w.max() * args.alpha)[0][-1]
        pred_timing = timing_index * 40 # in ms
        timing_index2 = numpy.where(att_w2 >= att_w2.max() * args.alpha)[0][-1]
        pred_timing2 = timing_index2 * 40 # in ms
    except:
        import pdb;pdb.set_trace()

    results[utt_id] = pred_timing2 - pred_timing

# Preview
print(f"avg is: {numpy.array([x for x in iter(results.values())]).mean()}")

alpha_str = str(args.alpha).replace('.','')
save_file_name = 'result_cmpDefault.pkl' if args.alpha == 1.0 else f'result_cmpDefault_alpha{alpha_str}.pkl'
with open(attn_dir / save_file_name, 'wb') as f:
    pickle.dump(results, f)
