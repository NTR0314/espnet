at="/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/force_alignment/mfa_eval2000_aligned"
bu="/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/inference/filter_bad_utts_eval/bad_utts.txt"
base_path="/export/data2/ozink/swbd/exp/asr_29_raw_en_bpe2000_sp/"
ms_time="500ms"

for file in ../29*0\.*.sh
do
    tmp=${base_path%%_raw*}
    num=${tmp##*asr_}
    tmp=${file##*Block_}
    factor=${tmp%%.sh}
    attn_path="${base_path}${ms_time}_${factor}_asr_model_valid.acc.ave/eval2000/attn_dir"

    echo $attn_path
    echo

    python3 "1_calc_preds_for_hist.py" $attn_path $at $bu && python3 "2_hist.py" $attn_path "${num}_${ms_time}_${factor}.png" &
done
