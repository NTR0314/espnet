# OSWALD: THIS SCRIPT NEEDS TO BE RUN FROM THE BASEDIR $SWBD

if [[ $# -lt 3 ]]; then
    echo "arg1 base false, arg2 gpu 1 arg3 gpu2"
    echo "THIS SCRIPT NEEDS TO BE RUN FROM THE BASEDIR swbd/libri"
    exit
fi

for run_file in $1
do
    for time_ms in 0 100 200 300 400 500 600 700 800 900 1000
    do
        tgt_path="./conf/${time_ms}ms_onlyMasked.yaml"
        cp ./conf/template_onlyMasked.yaml ${tgt_path}
        let blocks="$time_ms / 10"
        sed -i -r -e "s/TEMPLATE/${blocks}/" $tgt_path

        base_name=$(basename $run_file)
        postfix=${base_name##*_500_}
        prefix=${base_name%%_500_*}

        tgt_path="${prefix}_${time_ms}_${postfix}"
        cp $run_file $tgt_path
        sed -i -r -e "s/500ms_onlyMasked.yaml/${time_ms}ms_onlyMasked.yaml/" $tgt_path
        if [[ $time_ms -ge 600 ]]; then
            bash "$tgt_path" --stage 12 --stop_stage 12 --gpu_id $3 &
        else
            bash "$tgt_path" --stage 12 --stop_stage 12 --gpu_id $2 &
        fi
    done
done

