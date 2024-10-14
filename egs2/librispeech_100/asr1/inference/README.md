#### Contents
This folder contains scripts used during and after inference such as:
- plotting
- inference automation scripts

#### Examples:
For each script an example is provided to demonstate the usage.
`s13_rm_unmasked_tokens.py`:
```
This script is used internally by mWER automation scripts to generate a text file with removed unmasked tokens.
```

`auto_plots.sh`:
```
bash auto_plots.sh ../exp/asr_63_raw_en_bpe2000_sp/ 0.01 devAlpha yesDev
```

`plot_eos_eoa_swbd_dev_test.py`:
- This script generates the plot for the EOU EOA differences.
```
python3 plot_eos_eoa_swbd_dev_test.py
```

`auto_inf_maskedOnlyWER_nbest.sh`:
- Calculates nbest WER
- Uses `s13_rm_unmasked_tokens.py` under the hood.
```
bash inference/auto_inf_maskedOnlyWER_nbest.sh normal_de
coding_no_masking_default_model_neu_500_onlyMasked.sh exp/asr_default_reference_model_causal_raw_en_bpe5000_sp/ 5 10best20beam
```

`plot_koba_cmp.py`
- Creates mWER, WER, timing plot discussed with Mr. Kobayashi Sensei.
- mWER files are manually created containing 0ms-500ms results
```
python3 inference/plot_koba_cmp.py -bp exp/asr_default_reference_model_causal_raw_en_bpe5000_sp/ -bp2 exp/asr_26_raw_en_bpe5000_sp/ --figname jetzt_passt_alles_with_0 --alpha 01 mWER_default_1best_1beam.txt mWER_default_5best_20beam.txt mWER_default_10best_20beam.txt mWER_masked_1best_1beam.txt mWER_masked_5best_20beam.txt mWER_masked_10best_20beam.txt
```
- Example for .txt file in this context:
```
(espnet) [ozink asr1] cat mWER_default_10best_20beam.txt
0
6.5
18.3
19.3
59.0
70.0
```

`auto_inf_maskedOnly_script.sh`:
- Automate the inference process for multiple masking times
```
bash inference/auto_inf_maskedOnly_script.sh 61_60fixed_trainWith SilenceMask.sh 0 1 yesSWBD 0 20 10 fixSry yesPrint
```

`auto_inf_maskedOnlyWER.sh`:
- Evaluates the WER of the inferences.
```
bash inference/auto_inf_maskedOnlyWER.sh 63_29_neu_500_onlyMasked.sh exp/asr_63_raw_en_bpe2000_sp/ SWBD
```

`1_calc_preds_for_hist.py`:
- Used internally, e.g., in auto_plots.sh
