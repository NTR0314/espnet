# This script is used to automate the inference process for inference using differnt masking times 
# The script is made to generate scripts of timing 0, 100, ..., 1000ms
# The script is used both for SWBD and LS and various other options


if [[ $# -lt 3 ]]; then
    echo "WARNING: This script needs to be run directly from swbd/asr1 or libri/asr1 dir because it has relative paths to conf/tuning"
    echo
    echo "arg1 runfile: 31_26_005ctc.sh"
    echo "arg2 gpu1"
    echo "arg3 gpu2"
    echo "optional arg4: Use SWBD? Empty for LS100, e.g., \"\" (Switches path to GT data accordingly)"
    echo "optional arg5 ctc weight for inference if set sets the ctc weight from 0.3 to given value"
    echo "optional arg6 beam_size for inference. Default=1"
    echo "optional arg7 n_best for inference. Default=1"
    echo "optional arg8 suffix. Default no suffix"
    echo "optional arg9: Flag for just printing commands (used because high beamsize uses a lot of gpu memory -> hard to split automatically)"
    exit
fi

if [[ -n "$4" ]]
then
    echo "Using SWBD dataset"
fi

if [[ -n $6 ]]; then
    beam_size=$6
else
    beam_size="1"
    n_best="1"
fi
if [[ -n $5 ]]; then
    ctc_inf_weight=$5
else
    ctc_inf_weight="0.3"
fi

if [[ -n $7 ]]
then
    n_best=$7
fi

if [[ -n $8 ]]
then
    suffix=$8
fi

if [[ -n "$9" ]]
then
    echo "Just printing commands"
fi


echo "Using beam_size: $beam_size"
echo "Using CTC weight: $ctc_inf_weight"
echo "Using n_best: $n_best"
echo "Using suffix: $suffix"

for run_file in $1
do
    for time_ms in 0 100 200 300 400 500 600 700 800 900 1000
    do
        # Create yaml files in conf
        if [[ -n $suffix ]]
        then
            tgt_path="./conf/${time_ms}ms_onlyMasked_${suffix}.yaml"
        else
            tgt_path="./conf/${time_ms}ms_onlyMasked.yaml"
        fi
        cp ./conf/template_onlyMasked.yaml ${tgt_path}
        let blocks="$time_ms / 10"
        sed -i -r -e "s/TEMPLATE/${blocks}/" $tgt_path
        sed -i -r -e "s/ctc_weight: 0.3/ctc_weight: ${ctc_inf_weight}/" $tgt_path
        sed -i -re "s/beam_size: 1/beam_size: ${beam_size}/" $tgt_path
        sed -i -re "s/nbest: 5/nbest: ${n_best}/" $tgt_path

        base_name=$(basename $run_file)
        postfix="onlyMasked.sh"
        prefix=${base_name%%.sh}

        # Create run sh files for stage 12
        tgt_path="${prefix}_${time_ms}_${postfix}"
        cp $run_file $tgt_path
        if [[ -n $suffix ]]
        then
            sed -i -r -e "s/[0-9]+ms.yaml/${time_ms}ms_onlyMasked_${suffix}.yaml/" $tgt_path
        else
            sed -i -r -e "s/[0-9]+ms.yaml/${time_ms}ms_onlyMasked.yaml/" $tgt_path
        fi

        if [[ -n "$4" ]]
        then
            sed -i '31i\    --decode_only_masked_swbd true \\' $tgt_path
        else
            sed -i '31i\    --decode_only_masked true \\' $tgt_path
        fi

        if [[ -n "$8" ]]
        then
            echo bash "$tgt_path" --stage 12 --stop_stage 12 --gpu_id \<TODO\> &
        else
            if [[ $time_ms -ge 600 ]]; then
                bash "$tgt_path" --stage 12 --stop_stage 12 --gpu_id $3 &
            else
                bash "$tgt_path" --stage 12 --stop_stage 12 --gpu_id $2 &
            fi
        fi
    done
done

