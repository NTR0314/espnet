bad_utts = []
with open("../../dump/raw/eval2000/text") as f:
    lines = [x.strip() for x in f.readlines()]
    for line in lines:
        uid, ts = line.split(maxsplit=1)
        ws = ts.split()
        if len(ws) < 4 or r'[' in ts:
            bad_utts.append(uid)

for utt in bad_utts:
    print(utt)
