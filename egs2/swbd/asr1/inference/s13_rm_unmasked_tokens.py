import argparse
from pathlib import Path
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('hyp_path')
parser.add_argument('ref_path')
parser.add_argument('inference_blocks')
args = parser.parse_args()
hyp_path = Path(args.hyp_path)
ref_path = Path(args.ref_path)
inference_blocks = args.inference_blocks

mfa_libri_path = "/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/force_alignment/single_file_mfa_swbd.txt"

with open(ref_path) as f:
    ref_lines = [x.strip() for x in f.readlines()]
with open(hyp_path) as f:
    hyp_lines = [x.strip() for x in f.readlines()]

with open(mfa_libri_path) as f:
    lines = f.readlines()

def process(data_in):
    """ example: utt_id "w1,w2,,w4" "t1,t2,t3,t4" """
    # Manchmal ist im word vom timing auch spaces -> maxsplit = 1
    uid, words, timings = data_in.rstrip('\n').split()
    ws = words.split(',')
    timings = timings.strip("\"")
    ts = timings.split(',')
    assert len(ws) == len(ts)
    d = [(x, y) for x, y in zip(ws,ts) if x != r'"' and x != '']
    d = np.array(d)
    return uid, d


p_lines = [process(x) for x in lines]

inference_blocks = int(inference_blocks)
inference_ms = inference_blocks * 10

masked_hyps = []
masked_refs = []
for hyp_line, ref_line, (uid, d) in zip(hyp_lines, ref_lines, p_lines):
    try:
        ref_uid, ref = ref_line.split(maxsplit=1)
        # OSWALD: idk how to make this more clean
        hyp_split = hyp_line.split(maxsplit=1)
        if len(hyp_split) == 1:
            hyp_uid = hyp_split[0]
            hyp = ''
        else:
            hyp_uid, hyp = hyp_line.split(maxsplit=1)
    except:
        import pdb;pdb.set_trace()

    assert ref_uid == hyp_uid == uid, (ref_uid, hyp_uid, uid)

    masking_threshold = float(d[-1, 1]) * 1000 - float(inference_ms)
    masked_words = []
    non_masked_words = []

    for d_w, d_t in d:
        # d_t from s in ms
        d_t = float(d_t)
        d_t *= 1000
        if d_t <= masking_threshold:
            non_masked_words.append(d_w)
        else:
            masked_words.append(d_w)
    assert len(masked_words) + len(non_masked_words) == len(ref.split()), (masking_threshold, d, masked_words, non_masked_words, ref.split())
    cutoff_index = len(non_masked_words)
    masked_ref = ref.split()[cutoff_index:]
    masked_refs.append(uid + " " + " ".join(masked_ref) + '\n')
    assert masked_words == masked_ref, (masked_words, masked_ref)
    masked_hyp = hyp.split()[cutoff_index:]
    masked_hyps.append(uid + " " + " ".join(masked_hyp) + '\n')

# Saving files
with open(ref_path.parent / f"text_{inference_ms}", "w+") as f:
    f.writelines(masked_refs)
with open(hyp_path.parent / f"text_{inference_ms}", "w+") as f:
    f.writelines(masked_hyps)
