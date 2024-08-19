import os
import textgrid
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

    spans = timing_dict[utt_id]
    for i, timing_sw in enumerate(spans, start=1):
        enc_len = attn_data.shape[0]
        a = timing_sw.minTime * 1000 / 40 / enc_len
        a = 1 - a
        b = timing_sw.maxTime * 1000 / 40 / enc_len
        b = 1 - b
        c = timing_sw.mark
        if i == len(spans):
            eos_enc_block = (1 - b) * enc_len
            ax.axhline(eos_enc_block)
        if i % 2 == 0:
            ax.axvline(0, a, b, label = c, linewidth=4)
        else:
            ax.axvline(0.25, a , b ,label=c, linewidth=4, c='red')

        if i % 2 == 0:
            ax.axvline(0, a, b, label = c, linewidth=4)
        else:
            ax.axvline(0.25, a , b ,label=c, linewidth=4, c='red')

    # Plot mask part starting from EOS
    start_span_mask = int(eos_enc_block - (blocks_inference / 4)) - 0.499
    end_span_mask = eos_enc_block - 1 + 0.499
    ax.axhspan(start_span_mask, end_span_mask, facecolor="None", hatch="/", edgecolor="white")


    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.clf()
    plt.close(fig)

parser = argparse.ArgumentParser()
parser.add_argument('full_attn_path')
parser.add_argument('mfa_dir')
args = parser.parse_args()
attn_dir = Path(args.full_attn_path)
mfa_eval_path = args.mfa_dir
blocks_inference = open(attn_dir / 'blocks_inference.txt').read()
blocks_inference = int(blocks_inference)

# MFA timings
mfa_files = os.listdir(mfa_eval_path)
timing_dict = {}
for file in mfa_files:
    if ".TextGrid" not in file:
        print(f"Skipping file {file}")
        continue
    utt_id = file.split(".", maxsplit=1)[0]
    tg = textgrid.TextGrid.fromFile(Path(mfa_eval_path) / file)
    spans = [x for x in tg[0] if x.mark != '']
    timing_dict[utt_id] = spans


utt_folders = [x for x in os.listdir(attn_dir) if os.path.isdir(attn_dir / x)]
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
        # att_lis: list over decoder steps, each file in 1 x 4 x 1 x E
        concated = reduce(lambda x, y: numpy.concatenate((x, y), axis=0), att_list)
        # D x 1 x e
        concated_avg_head = numpy.average(concated, axis=1).squeeze(-2)
        # ENC x DEC
        try:
            concated_avg_head = numpy.swapaxes(concated_avg_head, 0, 1)
        except:
            import pdb;pdb.set_trace()

        filename = f"layer_{layer_str}_avg_head_concat.png"
        title_str = f"layer {layer_str} averaged over heads"
        save_path = attn_dir / utt_folder / filename
        # Plot each layer averaged over heads
        print(f"Plotting file: {save_path}")
        plot(concated_avg_head, title_str, save_path, blocks_inference = blocks_inference, utt_id=utt_folder)

        # For each layer plot each head.
        if False:
            for head, head_data in enumerate(concated):
                filename = f"layer_{layer_str}_head_{head}.png"
                title_str = f"layer {layer_str} head {head}.png"
                save_path = attn_dir / utt_folder / filename
                print(f"Plotting file: {save_path}")
                head_data = numpy.swapaxes(head_data, 0, 1)
                plot(head_data, title_str, save_path, blocks_inference = blocks_inference, utt_id=utt_folder)
