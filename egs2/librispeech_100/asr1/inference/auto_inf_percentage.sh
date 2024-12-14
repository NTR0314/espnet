if [[ $# -lt 3 ]]; then
    echo "This script needs to be run from base dir: swbd/libri"
    echo "mandatory arg1: run sh file: e.g., 31_26_005ctc.sh"
    echo "mandatory arg2 gpu_id 1"
    echo "mandatory arg3: gpu_id 2"
    echo "optional arg4: beam_size"
    echo "optional arg5: suffix (for own results in exp folder)"
    echo "optional arg6: Use dev set? Any non null string will change from test to dev"
    echo "optional arg7: Use SWBD dataset instead (has different names for dev set)"
    exit
fi

if [[ -n $4 ]]
then
    beam_size=$4
    if [[ $4 != 1 ]]
    then
        echo "explicitly setting beam_size sets ctc_weight to 0"
        ctc_weight=0
    else
        ctc_weight=0.3
    fi
else
    beam_size=1
    ctc_weight=0.3
fi

if [[ -n "$5" ]]
then
    echo "Using suffix: ${5}"
fi

if [[ -n "$6" ]]
then
    echo "Using dev set!"
fi

if [[ -n "$7" ]]
then
    echo "Using SWBD dataset instead of default LS100"
fi

echo "Using beam_size = ${beam_size}"
echo "Using CTC weight: $ctc_weight"

for mask_percentage in 0 10 20 30 40 50 60 70 80 90 100
do
    # conf YAMLs
    if [[ -n $5 ]]
    then
        tgt_path="./conf/${mask_percentage}_mask_${5}.yaml"
    else
        tgt_path="./conf/${mask_percentage}_mask.yaml"
    fi

    if [[ -n "$6" ]]
    then
        cp ./conf/template_dev.yaml ${tgt_path}
    else
        cp ./conf/template.yaml ${tgt_path}
    fi

    let blocks="9999"
    sed -i -r -e "s/TEMPLATE/${blocks}/" $tgt_path
    sed -i -re "s/beam_size: 1/beam_size: ${beam_size}/" $tgt_path
    sed -i -r -e "s/ctc_weight: 0.3/ctc_weight: ${ctc_weight}/" $tgt_path
    # Add percentage of mask of last word to config -> lwmp
    sed -i -re "/blocks_inference/a\\lwmp: ${mask_percentage}" $tgt_path

    base_path=${1%.sh}
    tgt_path="${base_path}_${mask_percentage}_lastWordMask.sh"
    cp $1 $tgt_path

    # Change test set
    if [[ -n "$6" ]]
    then
        if [[ -n "$7" ]]
        then
            sed -i -r -e "s/test_sets=\"eval2000\"/test_sets=\"train_dev\"/" $tgt_path
        else
            sed -i -r -e "s/test_sets=\"test_clean\"/test_sets=\"dev\"/" $tgt_path
        fi
    fi

    # Change conf lines
    if [[ -n $5 ]]
    then
        sed -i -r -e "s/[0-9]+ms.yaml/${mask_percentage}_mask_${5}.yaml/" $tgt_path
    else
        sed -i -r -e "s/[0-9]+ms.yaml/${mask_percentage}_mask.yaml/" $tgt_path
    fi

    if [[ $mask_percentage -ge 60 ]]; then
        bash "$tgt_path" --stage 12 --stop_stage 13 --gpu_id $3 &
    else
        bash "$tgt_path" --stage 12 --stop_stage 13 --gpu_id $2 &
    fi
done