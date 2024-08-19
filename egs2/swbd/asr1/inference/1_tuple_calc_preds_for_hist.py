import pickle
import textgrid
from pathlib import Path
import numpy
import argparse
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('attn_path', help="path to tuple dumps")
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

utt_folders = [x for x in os.listdir(attn_dir) if os.path.isdir(attn_dir / x)]
for utt_folder in utt_folders:
    # Filte bad utts
    #import pdb;pdb.set_trace()
    if utt_folder in bad_utts:
        print(f"skipped bad utt: {utt_folder}")
        continue

    filenames = os.listdir(attn_dir / utt_folder)
    filenames.sort()

    timing = numpy.load(attn_dir / utt_folder / filenames[-1], allow_pickle=True)
    pred_timing = timing.item()['decoder'].item() * 10

    actual_mfa_timing = timing_dict[utt_folder]
    print(actual_mfa_timing, pred_timing)
    results[utt_folder] = actual_mfa_timing - pred_timing


# Preview
for key in list(results.keys())[-10:]:
    print(f"{key=}")
    print(f"{results[key]=}")

with open(attn_dir / 'result.pkl', 'wb') as f:
    pickle.dump(results, f)
