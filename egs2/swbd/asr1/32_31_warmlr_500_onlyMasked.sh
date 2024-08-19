#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="eval2000"

asr_config=conf/tuning/32.yaml
inference_config="conf/500ms_onlyMasked.yaml"

./asr.sh \
    --lang en \
    --ngpu 5 \
    --token_type bpe \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 1 \
    --nbpe 2000 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --use_swbd_timings true \
    --decode_only_masked_swbd true \
    --bpe_train_text "data/${train_set}/text" "$@" \
