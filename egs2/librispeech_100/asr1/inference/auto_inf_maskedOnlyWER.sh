# This file automates the calculation of mWER. It internally uses the `s13_rm_unmasked_tokens.py` script.
# Internally it creates two text files that have only masked or partially masked tokens in them for both the hypothesis
# and the reference.

# These files are then renamed and used by the default ESPNET stage 13 WER calculation.

if [[ -z $1 ]]; then
    echo "arg1 run file"
    echo "Should be something like: 26_noExtraLossFull_500_decode_onlyMasked.sh"
    echo "IMPORTANT: This is the _500_ files not the default .sh"
    echo "arg2 base"
    echo "/export/data2/ozink/swbd/exp/asr_32_raw_en_bpe2000_sp/"
    echo "optional arg3: use SWBD instead of LIBRI"
    exit
fi
if [[ -z $2 ]]; then
    echo "arg2 base"
    echo "/export/data2/ozink/swbd/exp/asr_32_raw_en_bpe2000_sp/"
    exit
fi

if [[ -n "$3" ]]
then
    echo "Using SWBD dataset"
fi

for masking_time in 0 100 200 300 400 500
do
    # Adjust text files for hyp and ref (hyp based on arg2)
    if [[ -n "$3" ]]
    then
        python_path="/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/inference/s13_rm_unmasked_tokens.py"
        hyp_path="${2}${masking_time}ms_onlyMasked_asr_model_valid.acc.ave/eval2000/text"
        ref_path="/export/data2/ozink/raw/eval2000/text"
        masking_blocks=$((masking_time / 10))
        python3 ${python_path} --hyp_path ${hyp_path} --ref_path ${ref_path} --inference_blocks ${masking_blocks} --SWBD
    else
        python_path="/project/OML/master_theses/ozink/Waseda/espnet/egs2/librispeech_100/asr1/inference/s13_rm_unmasked_tokens.py"
        hyp_path="${2}${masking_time}ms_onlyMasked_asr_model_valid.acc.ave/test_clean/text"
        ref_path="/export/data2/ozink/librispeech_100/raw/test_clean/text"
        masking_blocks=$((masking_time / 10))
        python3 ${python_path} --hyp_path ${hyp_path} --ref_path ${ref_path} --inference_blocks ${masking_blocks}
    fi

    ref_base=${ref_path%/text}
    cp $ref_path "${ref_base}/text.backup"
    mv "${ref_base}/text_${masking_time}" "${ref_base}/text"

    hyp_base=${hyp_path%/text}
    cp $hyp_path "${hyp_base}/text.backup"
    mv "${hyp_base}/text_${masking_time}" "${hyp_base}/text"

    # 26_noExtraLossFull_500_decode_onlyMasked.sh
    dir_name=$(dirname $1)
    base_name=$(basename $1)
    postfix=${base_name##*_500_}
    prefix=${base_name%%_500_*}

    bash "${dir_name}/${prefix}_${masking_time}_${postfix}" --stage 13 --stop_stage 13

    mv "${hyp_base}/text.backup" $hyp_path 
    mv "${ref_base}/text.backup" $ref_path 
done
