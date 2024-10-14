# This script uses the scripts `1_calc_preds_for_hist.py` and `2_hist.py` to automatically calculate and generate histograms from the timing predictions
if [ -z $1 ]
then
    echo "arg1: basepath"
    echo "Basepath is the folder where all the inference_folders are store, e.g., exp/asr_xx_.../"
    echo "This script needs to be called from the inference/ folder"
    echo "optional: arg2: alpha parameter for koba decoding"
    echo "optional arg3: prefix, e.g., 100ms_devAlpha_..."
    echo "optional arg4: if it is dev set -> different MFA path"
    exit
fi

if [ -n "$2" ]
then
    echo "using alpha = $2"
    alpha=$2
fi

if [ -n "$3" ]
then
    echo "using prefix = $3"
fi

if [[ -n "$4" ]]
then
    echo " Using dev set MFA path "
fi

base_path=$1

if [[ -n "$4" ]]
then
    at="/project/OML/master_theses/ozink/Waseda/espnet/egs2/librispeech_100/asr1/force_alignments/mfa_dev"
else
    at="/project/OML/master_theses/ozink/Waseda/espnet/egs2/librispeech_100/asr1/force_alignments/mfa_libri/test-clean"
fi

for ms_time in 0 100 200 300 400 500 600 700 800 900 1000
do
    if [[ -n "$4" ]]
    then
        if [ -n "$3" ]
        then
            attn_path="${base_path}${ms_time}ms_${3}_asr_model_valid.acc.ave/org/dev/attn_dir"
        else
            attn_path="${base_path}${ms_time}ms_asr_model_valid.acc.ave/org/dev/attn_dir"
        fi
    else
        if [ -n "$3" ]
        then
            attn_path="${base_path}${ms_time}ms_${3}_asr_model_valid.acc.ave/test_clean/attn_dir"
        else
            attn_path="${base_path}${ms_time}ms_asr_model_valid.acc.ave/test_clean/attn_dir"
        fi
    fi

    tmp=${base_path%%_raw*}
    num=${tmp##*asr_}
    if [ -n "$2" ]
    then
        python3 "1_calc_preds_for_hist.py" $attn_path $at "--alpha" "${2}" && python3 "2_hist.py" $attn_path "${num}_${ms_time}msInf.png" &
    else
        python3 "1_calc_preds_for_hist.py" $attn_path $at && python3 "2_hist.py" $attn_path "${num}_${ms_time}msInf.png" &
    fi
done
