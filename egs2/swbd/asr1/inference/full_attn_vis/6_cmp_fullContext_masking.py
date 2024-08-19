import textgrid
import os
import argparse
from pathlib import Path
import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot(att1, att2, title, save_path, utt_id, bi_mask=0):
    # figsize: width, height
    # subplots arg0, arg1: rows, cols
    fig, axes = plt.subplots(1, 3, figsize=(45,15))
    #print(f"{bi_mask=}")
    axes[0].imshow(att1.astype(numpy.float32), aspect="auto")
    axes[0].set_title('full context')
    axes[1].imshow(att2.astype(numpy.float32), aspect="auto")
    axes[1].set_title('masked context')
    axes[2].imshow(numpy.abs(att1[:, :att2.shape[1]] - att2).astype(numpy.float32), aspect="auto")
    axes[2].set_title('abs diff of attn')
    fig.suptitle(f"utt : {str(title)}")
    for ax in axes:
        ax.set_xlabel("Decoder steps")
        ax.set_ylabel("Encoder steps")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


    # Plot all word timings
    if utt_id not in timing_dict:
        return
    spans = timing_dict[utt_id]
    for att, ax in zip([att1, att2], axes):
        enc_len = att.shape[0]
        for i, timing_sw in enumerate(spans, start=1):
            a = timing_sw.minTime * 1000 / 40 / enc_len
            a = 1 - a
            b = timing_sw.maxTime * 1000 / 40 / enc_len
            b = 1 - b
            c = timing_sw.mark
            if i % 2 == 0:
                ax.axvline(0, a, b, label = c, linewidth=4)
            else:
                ax.axvline(0.25, a , b ,label=c, linewidth=4, c='red')

    ax = axes[1]
    last_end_timing = (1 - b) * enc_len
    start_span = max(0, int(last_end_timing - (bi_mask / 4)) - 0.499) # max(0, ..) because masking can be longer than audio/enc len. Is capped in the same way in fw pass
    end_span = last_end_timing - 1 + 0.499
    # if "030185-03024" in utt_id:
    #     import pdb;pdb.set_trace()
    ax.axhspan(start_span, end_span, facecolor="None", hatch="/", edgecolor="white")
    axes[2].axhspan(start_span, end_span, facecolor="None", hatch="/", edgecolor="white")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.clf()
    plt.close(fig)

parser = argparse.ArgumentParser()
parser.add_argument('attn_dir_1', help="FULL CONTEXT")
parser.add_argument('attn_dir_2', help="MASKED CONTEXT")
parser.add_argument('mfa', help="mfa dir eval200")
args = parser.parse_args()
attn_dir_1 = Path(args.attn_dir_1)
attn_dir_2 = Path(args.attn_dir_2)
blocks_inference_1 = int(open(attn_dir_1 / 'blocks_inference.txt').read())
blocks_inference_2 = int(open(attn_dir_2 / 'blocks_inference.txt').read())
assert blocks_inference_1 != blocks_inference_2
assert blocks_inference_1 == 0

# MFA timings
mfa_eval_path = args.mfa
mfa_files = os.listdir(mfa_eval_path)
timing_dict = {}
for file in mfa_files:
    if ".TextGrid" not in file:
        print(f"Skipping file {file}")
        continue
    utt_id = file.split(".", maxsplit=1)[0]
    tg = textgrid.TextGrid.fromFile(Path(mfa_eval_path) / file)
    spans = [x for x in tg[0] if x.mark != '']
    # last_timing = spans[-1].maxTime * 1000 # in ms
    timing_dict[utt_id] = spans

utt_folders_1 = [x for x in os.listdir(attn_dir_1) if "blocks_inference" not in x]
utt_folders_2 = [x for x in os.listdir(attn_dir_2) if "blocks_inference" not in x]
utt_folders_1.sort()
utt_folders_2.sort()
for u1, u2 in zip(utt_folders_1, utt_folders_2):
    assert u1 == u2, (u1, u2)
    u = u1
    att_dumps_1, att_dumps_2 = {}, {}
    fs1 = os.listdir(attn_dir_1 / u1)
    fs2 = os.listdir(attn_dir_2 / u2)
    fs1.sort()
    fs2.sort()

    for files, attn_dir, att_dumps in [(fs1, attn_dir_1, att_dumps_1), (fs2, attn_dir_2, att_dumps_2)]:
        for file in files:
        # ignore non numpy files
            if not ".npy" in file:
                continue
            fp = attn_dir / u / file
            att_w = numpy.load(fp)
            layer = 6
            if str(layer) not in att_dumps:
                att_dumps[str(layer)] = [att_w]
            else:
                att_dumps[str(layer)].append(att_w)

    for (l1, a1), (l2, a2) in zip(att_dumps_1.items(), att_dumps_2.items()):
        from functools import reduce
        c1 = reduce(lambda x, y: numpy.concatenate((x, y), axis=2), a1)
        c2 = reduce(lambda x, y: numpy.concatenate((x, y), axis=2), a2)
        c_avg_1 = numpy.average(c1, axis=1).squeeze(0)
        c_avg_2 = numpy.average(c2, axis=1).squeeze(0)
        c_avg_1 = numpy.swapaxes(c_avg_1, 0, 1)
        c_avg_2 = numpy.swapaxes(c_avg_2, 0, 1)
        assert l1 == l2, (l1, l2)
        filename = f"layer_{l1}_avg_head_comp_FullContext_MaskedContext.png"
        title_str = filename[:-4]
        # Save in masked dir. 
        save_path = attn_dir_2 / u2 / filename
        print(f"Plotting file: {save_path}")
        plot(c_avg_1, c_avg_2, title_str, save_path, bi_mask = blocks_inference_2, utt_id=u1)

