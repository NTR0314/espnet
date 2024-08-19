: '
positional arguments:
    attn_path        Path to attn_dir where all the numpyziz are saved
    at               Path of MFA timing folder
    bu               bad utt txt file
'

: '
usage: 2_hist.py [-h] attn_path filename_plot

positional arguments:
  attn_path
    filename_plot
'

if [ -z $1 ]
then
    echo "damn bwoah you need to set the basepath 1"
    exit
fi
if [ -z $2 ]
then
    echo "damn bwoah you need to set the basepath 2"
    exit
fi

base_path=$1
base_path2=$2
at="/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/force_alignment/mfa_eval2000_aligned"
bu="/project/OML/master_theses/ozink/Waseda/espnet/egs2/swbd/asr1/inference/filter_bad_utts_eval/bad_utts.txt"

for ms_time in 0 100 200 300 400 500 600 700 800 900 1000
do
    attn_path="${base_path}${ms_time}ms_asr_model_valid.acc.ave/eval2000/attn_dir"
    attn_path_2="${base_path2}${ms_time}ms_asr_model_valid.acc.ave/eval2000/attn_dir"
    tmp=${base_path%%_raw*}
    num=${tmp##*asr_}
    python3 "3_overlapping_2hist.py" $attn_path $attn_path_2 "${num}_${ms_time}ms_comp.png"
done
