import pickle
import textgrid
from pathlib import Path
import numpy
import argparse
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('attn_path', help="Path to attn_dir where all the numpyziz are saved")
parser.add_argument('at', help='Path of MFA timing folder')
parser.add_argument('bu', help='bad utt txt file path')
parser.add_argument('--use_last_head', action='store_true', help="flag if should use last head for argmax, else will use average")
args = parser.parse_args()
attn_dir = Path(args.attn_path)
results = {}

print(f"{args.use_last_head=}")

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

for att_file_name in os.listdir(attn_dir):
    if not ".npy" in att_file_name:
        continue

    # Get utt_id from filename
    match = re.search('(sw|en)_\d{4}-[AB]{1}_(?P<start>\d{6})-(?P<end>\d{6})', att_file_name)
    utt_id = match.group()
    # Filte bad utts
    if utt_id in bad_utts:
        print(f"skipped bad utt: {utt_id}")
        continue

    att_w = numpy.load(attn_dir / att_file_name)
    if att_w.ndim == 2:
        att_w = att_w[None]
    elif att_w.ndim == 4:
    # In multispkr_asr model case, the dimension could be 4.
        # B x H x D x E -> BH x D x E == H x D x E
        att_w = numpy.concatenate([att_w[i] for i in range(att_w.shape[0])], axis=0)
    elif att_w.ndim > 4 or att_w.ndim == 1:
        raise RuntimeError(f"Must be 2, 3 or 4 dimension: {att_w.ndim}")
    # remove d = 1 dim
    att_w = att_w.squeeze()

    # Take avg/last head for inference
    if not args.use_last_head:
        att_w = numpy.average(att_w, axis=0)
    else:
        att_w = att_w[-1]
    argmax_att = att_w.argmax()
    pred_timing = argmax_att * 40 # in ms
    actual_mfa_timing = timing_dict[utt_id]
    results[utt_id] = actual_mfa_timing - pred_timing


# Preview
for key in list(results.keys())[-10:]:
    print(f"{key=}")
    print(f"{results[key]=}")

with open(attn_dir / 'result.pkl', 'wb') as f:
    pickle.dump(results, f)
