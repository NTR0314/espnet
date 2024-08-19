# for masking_time in 0 100 200 300 400 500 600 700 800 900 1000
if [[ -z $1 ]]; then
    echo "arg1 run file"
    echo "Should be something like: 26_noExtraLossFull_500_decode_onlyMasked.sh"
    exit
fi
if [[ -z $2 ]]; then
    echo "arg2 base"
    echo "/export/data2/ozink/swbd/exp/asr_32_raw_en_bpe2000_sp/"
    exit
fi


for masking_time in 0 100 200 300 400 500 600 700 800 900 1000
do
    python_path="/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/inference/s13_rm_unmasked_tokens.py"
    hyp_path="${2}${masking_time}ms_onlyMasked_asr_model_valid.acc.ave/eval2000/text"
    ref_path="/export/data2/ozink/raw/eval2000/text"
    masking_blocks=$((masking_time / 10))
    python3 ${python_path} ${hyp_path} ${ref_path} ${masking_blocks}

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
