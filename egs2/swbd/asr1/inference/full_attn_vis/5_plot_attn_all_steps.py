import textgrid
import sentencepiece as spm
import os
import argparse
from pathlib import Path
import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot(attn_data, title, save_path, utt_id, blocks_inference=0):
    fig, ax = plt.subplots(figsize=(15,15))
    plt.imshow(attn_data.astype(numpy.float32), aspect="auto")
    fig.suptitle(f"utt : {str(title)}")
    plt.xlabel("Decoder steps")
    plt.ylabel("Encoder steps")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Box for masked part
    start_span = int(attn_data.shape[0] - (blocks_inference / 4)) - 0.499
    end_span = attn_data.shape[0] - 1 + 0.499
    ax.axhspan(start_span, end_span, facecolor="None", hatch="/", edgecolor="white")

    # Plot all word timings
    if utt_id not in timing_dict:
        return
    spans = timing_dict[utt_id]
    for i, timing_sw in enumerate(spans, start=1):
        enc_len = attn_data.shape[0]
        a = timing_sw.minTime * 1000 / 40 / enc_len
        a = 1 - a
        b = timing_sw.maxTime * 1000 / 40 / enc_len
        b = 1 - b
        c = timing_sw.mark
        if i % 2 == 0:
            ax.axvline(0, a, b, label = c, linewidth=4)
        else:
            ax.axvline(0.25, a , b ,label=c, linewidth=4, c='red')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.clf()
    plt.close(fig)

parser = argparse.ArgumentParser()
parser.add_argument('attn_dir_full', help="attn dir of data where all dec steps are dumped")
parser.add_argument('at', help="path to mfa Folder")
args = parser.parse_args()
attn_dir = Path(args.attn_dir_full)
blocks_inference = open(attn_dir / 'blocks_inference.txt').read()
blocks_inference = int(blocks_inference)

# MFA timings
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
    # last_timing = spans[-1].maxTime * 1000 # in ms
    timing_dict[utt_id] = spans

utt_folders = [x for x in os.listdir(attn_dir) if "blocks_inference" not in x]
utt_folders.sort()
for utt_folder in utt_folders:
    att_dumps = {}
    filenames = os.listdir(attn_dir / utt_folder)
    filenames.sort()

    for filename in filenames:
        # ignore non numpy files
        if not ".npy" in filename:
            continue
        full_path = attn_dir / utt_folder / filename
        att_w = numpy.load(full_path)
        layer = 6
        if str(layer) not in att_dumps:
            att_dumps[str(layer)] = [att_w]
        else:
            att_dumps[str(layer)].append(att_w)

    for layer_str, att_list in att_dumps.items():
        from functools import reduce
        # att_list is list of len decoder_steps and each element \in B(1) x H(4) x D(1) x E
        concated = reduce(lambda x, y: numpy.concatenate((x, y), axis=2), att_list).squeeze(0) #squeeze batch shape
        # H x D x E -> D x E
        concated_avg_head = numpy.average(concated, axis=0)
        # ENC x DEC
        concated_avg_head = numpy.swapaxes(concated_avg_head, 0, 1)
        filename = f"layer_{layer_str}_avg_head_concat.png"
        title_str = f"layer {layer_str} averaged over heads"
        save_path = attn_dir / utt_folder / filename
        # Plot each layer averaged over heads
        print(f"Plotting file: {save_path}")
        plot(concated_avg_head, title_str, save_path, blocks_inference = blocks_inference, utt_id=utt_folder)

        if False:
            for head, head_data in enumerate(concated):
                filename = f"layer_{layer_str}_head_{head}.png"
                title_str = f"layer {layer_str} head {head}.png"
                save_path = attn_dir / utt_folder / filename
                print(f"Plotting file: {save_path}")
                head_data = numpy.swapaxes(head_data, 0, 1)
                plot(head_data, title_str, save_path, blocks_inference = blocks_inference, utt_id=utt_folder)

