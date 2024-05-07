export CUDA_VISIBLE_DEVICES="0"

python3 -m espnet2.bin.asr_inference --batch_size 1 --ngpu 1 --data_path_and_name_and_type dump/raw/test_clean/wav.scp,speech,kaldi_ark --key_file exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/test_clean/logdir/keys.1.scp --asr_train_config exp/asr_train_asr_raw_en_bpe5000_sp/config.yaml --asr_model_file exp/asr_train_asr_raw_en_bpe5000_sp/valid.acc.ave.pth --output_dir exp/asr_train_asr_raw_en_bpe5000_sp/decode_asr_asr_model_valid.acc.ave/test_clean/logdir/output.1 --config conf/decode_asr.yaml
