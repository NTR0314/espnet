if [[ $# -lt 3 ]]; then
    echo "arg1 runfile: 31_26_005ctc.sh"
    echo "arg2 gpu1"
    echo "arg3 gpu2"
    echo "needs to be run from swbd/asr1 or libri/asr1 dir because it has relative paths to conf/tuning"
    echo "optional arg4 ctc weight for inference if set sets the ctc weight from 0.3 to given value"
    exit
fi

if [[ -n $4 ]]; then
    ctc_inf_weight=$4
    echo "Using CTC weight: $ctc_inf_weight"
else
    ctc_inf_weight="0.3"
fi


for run_file in $1
do
    for time_ms in 0 100 200 300 400 500 600 700 800 900 1000
    do
        # Create yaml files in conf
        tgt_path="./conf/${time_ms}ms_onlyMasked.yaml"
        cp ./conf/template_onlyMasked.yaml ${tgt_path}
        let blocks="$time_ms / 10"
        sed -i -r -e "s/TEMPLATE/${blocks}/" $tgt_path
        sed -i -r -e "s/ctc_weight: 0.3/ctc_weight: ${ctc_inf_weight}/" $tgt_path

        base_name=$(basename $run_file)
        postfix="onlyMasked.sh"
        prefix=${base_name%%.sh}

        # Create run sh files for stage 12
        tgt_path="${prefix}_${time_ms}_${postfix}"
        cp $run_file $tgt_path
        sed -i -r -e "s/500ms.yaml/${time_ms}ms_onlyMasked.yaml/" $tgt_path
        sed -i '31i\    --decode_only_masked true \\' $tgt_path
        if [[ $time_ms -ge 600 ]]; then
            bash "$tgt_path" --stage 12 --stop_stage 12 --gpu_id $3 &
        else
            bash "$tgt_path" --stage 12 --stop_stage 12 --gpu_id $2 &
        fi
    done
done

