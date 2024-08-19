if [ -z $1 ]
then
    echo "damn bwoah you need to set the basepath"
    echo "Basepath is the folder where all the inference_folders are store, e.g., exp/asr_xx_.../"
    echo "This script needs to be called from the inference/ folder"
    exit
fi

base_path=$1
at="/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/force_alignment/mfa_eval2000_aligned"
bu="/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/inference/filter_bad_utts_eval/bad_utts.txt"

for ms_time in 0 100 200 300 400 500 600 700 800 900 1000
do
    attn_path="${base_path}${ms_time}ms_asr_model_valid.acc.ave/eval2000/attn_dir"
    tmp=${base_path%%_raw*}
    num=${tmp##*asr_}
    python3 "1_calc_preds_for_hist.py" $attn_path $at $bu && python3 "2_hist.py" $attn_path "${num}_${ms_time}msInf.png" &
done
