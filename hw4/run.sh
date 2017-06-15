#!/bin/bash
wget -O works.zip "https://www.dropbox.com/s/xplxts5g9ousjs6/model.zip?dl=0"
unzip works.zip -d ./
python3 main.py --mode test --model_name "$1" --test_dataset_path "$2" --results_dir "$3"