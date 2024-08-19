if [ -z $1 ]
then
    echo "arg1: basepath"
    echo "Basepath is the folder where all the inference_folders are store, e.g., exp/asr_xx_.../"
    echo "This script needs to be called from the inference/ folder"
    echo "optional: arg2: alpha parameter for koba decoding"
    exit
fi

if [ -n "$2" ]
then
    echo "using alpha = $2"
    alpha=$2
fi

base_path=$1
at="/project/OML/master_theses/ozink/Waseda/espnet/egs2/librispeech_100/asr1/force_alignments/mfa_libri/test-clean"

for ms_time in 0 100 200 300 400 500 600 700 800 900 1000
do
    attn_path="${base_path}${ms_time}ms_asr_model_valid.acc.ave/test_clean/attn_dir"
    tmp=${base_path%%_raw*}
    num=${tmp##*asr_}
    if [ -n "$2" ]
    then
        # echo python3 "1_calc_preds_for_hist.py" $attn_path $at && echo python3 "2_hist.py" $attn_path "${num}_${ms_time}msInf.png" "--alpha" "${2}" &
        python3 "1_calc_preds_for_hist.py" $attn_path $at "--alpha" "${2}" && python3 "2_hist.py" $attn_path "${num}_${ms_time}msInf.png" &
    else
        # echo python3 "1_calc_preds_for_hist.py" $attn_path $at && echo python3 "2_hist.py" $attn_path "${num}_${ms_time}msInf.png" &
        python3 "1_calc_preds_for_hist.py" $attn_path $at && python3 "2_hist.py" $attn_path "${num}_${ms_time}msInf.png" &
    fi
done
