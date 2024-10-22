"""
This script is used for calculating mWER by removing words from the transcript text (can be either reference or hyp),
by using the MFA timings.
"""

import argparse
import textgrid
from pathlib import Path
import torch
import numpy as np
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('--ref_path', nargs=1, required=True)
parser.add_argument('--inference_blocks', nargs=1)
parser.add_argument('--hyp_path', nargs='+', required=True)
parser.add_argument('--calc_wer', action='store_true')
parser.add_argument('--num_n_best', default=5, type=int)
parser.add_argument('--SWBD', action='store_true')
args = parser.parse_args()
hyp_paths = args.hyp_path
ref_path = Path(args.ref_path[0])
inference_blocks = args.inference_blocks[0]
calc_wer = args.calc_wer

if not args.SWBD:
    mfa_libri_path = "/project/OML/master_theses/ozink/Waseda/espnet/egs2/librispeech_100/asr1/force_alignments/LibriSpeech/test-clean/mfa_test-clean.txt"
else:
    mfa_path = "/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/force_alignment/mfa_eval2000_aligned"
    # Load MFA timings
    mfa_files = os.listdir(mfa_path)
    timing_dict = {}
    for file in mfa_files:
        if ".TextGrid" not in file:
            print(f"Skipping file {file}")
            continue
        utt_id = file.split(".", maxsplit=1)[0]
        tg = textgrid.TextGrid.fromFile(Path(mfa_path) / file)
        spans = [x for x in tg[0] if x.mark != '']
        timing_dict[utt_id] = spans


def process(data_in):
    """ example: utt_id "w1,w2,,w4" "t1,t2,t3,t4" """
    uid, words, timings = data_in.rstrip('\n').split()
    ws = words.split(',')
    timings = timings.strip("\"")
    ts = timings.split(',')
    assert len(ws) == len(ts)
    d = [(x, y) for x, y in zip(ws,ts) if x != r'"' and x != '']
    d = np.array(d)
    return uid, d

inference_blocks = int(inference_blocks)
inference_ms = inference_blocks * 10

with open(ref_path) as f:
    ref_lines = [x.strip() for x in f.readlines()]

if not args.SWBD:
    with open(mfa_libri_path) as f:
        lines = f.readlines()
        p_lines = [process(x) for x in lines]

    hyp_lines_dict = {}
    masked_hyps_dict = {}
    for hyp_path in hyp_paths:
        with open(Path(hyp_path)) as f:
            re_match = re.search('(\d)best_recog', hyp_path)
            if re_match is not None:
                n_best = re_match.group(1)
            else:
                n_best = 1
            hyp_lines = [x.strip() for x in f.readlines()]

            # ensure that missing uids are added because sometimes there is no hyp especially for higher n_best
            # uids_ref = {line.split(maxsplit=1)[0] for line in ref_lines}
            # uids_n_best_hyp = {line.split(maxsplit=1)[0] for line in hyp_lines}
            # missing_uids = uids_ref - uids_n_best_hyp
            # missing_lines = [f"{uid}" for uid in missing_uids]
            # hyp_lines += missing_lines
            # hyp_lines.sort(key=lambda line: line.split(maxsplit=1)[0])

            hyp_lines_dict[int(n_best)] = hyp_lines

        masked_hyps = []
        masked_refs = []
        for hyp_line, ref_line, (uid, d) in zip(hyp_lines, ref_lines, p_lines):
            ref_uid, ref = ref_line.split(maxsplit=1)
            # TODO: Can be no token decoded -> .split() fails
            if ' ' in hyp_line:
                hyp_uid, hyp = hyp_line.split(maxsplit=1)
            else:
                hyp_uid = hyp_line
                hyp = ''

            try:
                assert ref_uid == hyp_uid == uid, (ref_uid, hyp_uid, uid, hyp_path)
            except:
                import pdb;pdb.set_trace()

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
        with open(Path(hyp_path).parent / f"text_{inference_ms}", "w+") as f:
            f.writelines(masked_hyps)
            masked_hyps_dict[int(n_best)] = masked_hyps

# SWBD
else:
    hyp_lines_dict = {}
    masked_hyps_dict = {}
    for hyp_path in hyp_paths:
        # Read hyp
        with open(Path(hyp_path)) as f:
            re_match = re.search('(\d)best_recog', hyp_path)
            if re_match is not None:
                n_best = re_match.group(1)
            else:
                n_best = 1
            hyp_lines = [x.strip() for x in f.readlines()]
            hyp_lines_dict[int(n_best)] = hyp_lines

        masked_hyps = []
        masked_refs = []
        for hyp_line, ref_line, in zip(hyp_lines, ref_lines):
            ref_uid, ref = ref_line.split(maxsplit=1)
            # TODO: Can be no token decoded -> .split() fails
            if ' ' in hyp_line:
                hyp_uid, hyp = hyp_line.split(maxsplit=1)
            else:
                hyp_uid = hyp_line
                hyp = ''

            try:
                assert ref_uid == hyp_uid, (ref_uid, hyp_uid, hyp_path)
                uid = hyp_uid
            except:
                import pdb;pdb.set_trace()

            masking_threshold = timing_dict[hyp_uid][-1].maxTime * 1000 - float(inference_ms)
            masked_words = []
            non_masked_words = []

            for d_w, d_t in [(x.mark, x.maxTime) for x in timing_dict[hyp_uid]]:
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
        with open(Path(hyp_path).parent / f"text_{inference_ms}", "w+") as f:
            f.writelines(masked_hyps)
            masked_hyps_dict[int(n_best)] = masked_hyps

if calc_wer:
    from jiwer import wer
    debug = False
    all_hyps = [y for x, y in masked_hyps_dict.items() if x <= args.num_n_best]
    refs = masked_refs
    wers=[]
    for combined in zip(refs, *all_hyps):
        uid, ref = combined[0].strip().split(maxsplit=1)
        if debug:
            print(ref)
        best_wer = 9999999
        for hyp_line in combined[1:]:
            hyp_line=hyp_line.strip()
            if ' ' in hyp_line:
                hyp_uid, hyp = hyp_line.split(maxsplit=1)
            else:
                hyp_uid = hyp_line
                hyp = ''
            hyp_wer = wer(ref, hyp)
            if debug:
                print(hyp_line)
                print(hyp_wer)
            if hyp_wer < best_wer:
                best_wer = hyp_wer
        wers.append(best_wer)
        if debug:
            print(best_wer)
    wers=np.array(wers)
    mean_wer=wers.mean() * 100
    print(f"{mean_wer:.1f}")
