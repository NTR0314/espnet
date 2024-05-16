import os
from pathlib import Path
import numpy
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('attn_path', help="Path to attn_dir where all the numpyziz are saved")
args = parser.parse_args()
attn_dir = Path(args.attn_path)
results = {}
for att_file_name in os.listdir(attn_dir):
    if ".pkl" in att_file_name:
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
    # avg_heads
    att_w = numpy.average(att_w, axis=0)
    argmax_att = att_w.argmax()
    utt_id = att_file_name.split('_')[1]
    # Length in ms
    results[utt_id] = argmax_att * 40


# save results
import pickle

# Preview
for key in list(results.keys())[-10:]:
    print(f"{key=}")
    print(f"{results[key]=}")

with open(attn_dir / 'result.pkl', 'wb') as f:
    pickle.dump(results, f)
