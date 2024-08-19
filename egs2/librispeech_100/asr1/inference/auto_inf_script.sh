if [[ $# -lt 3 ]]; then
    echo "arg1 run sh file: 31_26_005ctc.sh"
    echo "arg2 gpu 1 arg3 gpu2"
    echo "This scripts appends _xxxms to the arg1 runfile"
    exit
fi

for time_ms in 0 100 200 300 400 500 600 700 800 900 1000
do
    # conf YAMLs
    tgt_path="./conf/${time_ms}ms.yaml"
    cp ./conf/template.yaml ${tgt_path}
    let blocks="$time_ms / 10"
    sed -i -r -e "s/TEMPLATE/${blocks}/" $tgt_path

    base_path=${1%.sh}
    tgt_path="${base_path}_${time_ms}.sh"
    cp $1 $tgt_path
    sed -i -r -e "s/500ms.yaml/${time_ms}ms.yaml/" $tgt_path
    if [[ $time_ms -ge 600 ]]; then
        bash "$tgt_path" --stage 12 --stop_stage 13 --gpu_id $3 &
    else
        bash "$tgt_path" --stage 12 --stop_stage 13 --gpu_id $2 &
    fi
done
