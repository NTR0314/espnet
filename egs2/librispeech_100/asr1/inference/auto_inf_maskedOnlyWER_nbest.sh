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
if [[ -n $3 ]]; then
    echo "setting n_best as ${3}_best"
    n_best_num=$3
fi

for masking_time in 100 200 300 400 500 600 700 800 900 1000
#for masking_time in 0 # 100 200 300 400 500 600 700 800 900 1000
do
    # Adjust text files for hyp and ref (hyp based on arg2)
    python_path="/project/OML/master_theses/ozink/Waseda/espnet/egs2/librispeech_100/asr1/inference/s13_rm_unmasked_tokens.py"
    hyp_path="${2}${masking_time}ms_onlyMasked_asr_model_valid.acc.ave/test_clean/text"
    hyp_parent=$(dirname ${hyp_path})
    n_best_paths=$(find "${hyp_parent}/logdir/output.1" -type f -regex '.*/text')

    ref_path="/export/data2/ozink/librispeech_100/raw/test_clean/text"
    masking_blocks=$((masking_time / 10))

    for n_best_path in $n_best_paths
    do
        # echo $n_best_path
        n_best_parent=$(dirname $n_best_path)
        diff <(cut -d ' ' -f 1 "/project/OML/master_theses/ozink/Waseda/espnet/egs2/librispeech_100/asr1/dump/raw/test_clean/text") <(cut -d ' ' -f 1 $n_best_path ) | rg '<' | cut -f 2 -d ' ' >> "${n_best_parent}/empty_utts.txt"
        cat "${n_best_parent}/empty_utts.txt" $n_best_path > "${n_best_parent}/text_with_missing_utt_unsorted.txt"
        mv $n_best_path "${n_best_parent}/text.backup"
        sort "${n_best_parent}/text_with_missing_utt_unsorted.txt" > "${n_best_parent}/text" 
    done

    # refill empty lines since it is not guaranteed that all n_best amount of hyps end in beam_search

    echo python3 ${python_path} --hyp_path $n_best_paths --ref_path ${ref_path} --inference_blocks ${masking_blocks} --calc_wer --num_n_best $n_best_num
done
