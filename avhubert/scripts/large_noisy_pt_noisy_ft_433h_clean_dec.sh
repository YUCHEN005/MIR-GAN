#!/usr/bin/env bash

source activate CONDA_ENV

python -B infer_s2s.py --config-dir ../conf/av-finetune --config-name large_noisy_pt_noisy_ft_433h_clean_dec.yaml

