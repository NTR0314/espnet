import os
from pathlib import Path
import numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot(attn_data, title, save_path):
    fig, ax = plt.subplots(figsize=(15,15))
    plt.imshow(attn_data.astype(numpy.float32), aspect="auto")
    fig.suptitle(f"utt : {str(title)}")
    plt.xlabel("Decoder steps")
    plt.ylabel("Encoder steps")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.clf()
    plt.close(fig)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('attn_dir', help="attn_dir")
args = parser.parse_args()
attn_dir = Path(args.attn_dir)

att_dumps = {}
filenames = os.listdir(attn_dir)
filenames.sort()
for filename in filenames:
    # ignore non numpy files
    if not ".npy" in filename:
        continue
    full_path = attn_dir / filename
    att_w = numpy.load(full_path)

    # some shit from initial script
    if att_w.ndim == 2:
        att_w = att_w[None]
    elif att_w.ndim == 4:
    # In multispkr_asr model case, the dimension could be 4.
        att_w = numpy.concatenate([att_w[i] for i in range(att_w.shape[0])], axis=0)
    elif att_w.ndim > 4 or att_w.ndim == 1:
        raise RuntimeError(f"Must be 2, 3 or 4 dimension: {att_w.ndim}")

    # Get utt-id
    import re
    utt_id = re.search("(utt_\d+-\d+-\d+)_step_\d+.npy", filename).groups()[0]
    if str(utt_id) not in att_dumps:
        att_dumps[str(utt_id)] = att_w
    else:
        raise Error("Duplicate utt_id")

for utt_id, att_w in att_dumps.items():
    # remove 1 dimensional decoder size
    # import pdb; pdb.set_trace()
    # att_w = numpy.squeeze(att_w)
    concated_avg_head = numpy.average(att_w, axis=0)

    print(f"{concated_avg_head.shape=}")
    # ENC x DEC
    concated_avg_head = numpy.swapaxes(concated_avg_head, 0, 1)
    print(f"{concated_avg_head.shape=}")
    filename = f"{utt_id}_attn_vis.png"
    title_str = f"Last layer: {utt_id} avg heads"
    save_path = attn_dir / filename
    # Plot each layer averaged over heads
    plot(concated_avg_head, title_str, save_path)

