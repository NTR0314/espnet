# OSWALD: THIS SCRIPT NEEDS TO BE RUN FROM THE BASEDIR $SWBD
# "../19_distill_allHeads_allLayers_allSteps_0005_short.sh"
# arg1: base run_file, e.g. 19_.....
# arg2: gpu_id
# arg3: gpu_id 2

if [[ $# -lt 3 ]]; then
    echo "arg 1 base run dir, arg2/3 gpu ids"
fi

for run_file in $1
do
    for eos_factor in 0.0001 0.00016681 0.00027826 0.00046416 0.00077426 0.00129155 0.00215443 0.00359381 0.00599484 0.01      
    do
        tgt_path="./conf/500ms_${eos_factor}.yaml"
        cp ./conf/template_eos.yaml ${tgt_path}
        sed -i -r -e "s/TEMPLATE/${eos_factor}/" $tgt_path
        base_path=${run_file%.sh}
        tgt_path="${base_path}_${eos_factor}.sh"
        cp $run_file $tgt_path
        sed -i -r -e "s/500ms.yaml/500ms_${eos_factor}.yaml/" $tgt_path
        # check does not work
        if [[ $eos_factor -le 0.0013 ]]; then
            bash "$tgt_path" --stage 12 --stop_stage 13 --gpu_id $2 &
        else
            bash "$tgt_path" --stage 12 --stop_stage 13 --gpu_id $3 &
        fi
    done
done
