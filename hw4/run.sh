#!/bin/bash
wget -O works.zip "https://www.dropbox.com/s/v0dtiuchgtgnmkx/works.zip?dl=0"
unzip -u works.zip
python3 main.py --mode test --model_name "$1" --test_dataset_path "$2" --results_dir "$3"
