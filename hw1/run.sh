#!/bin/bash
wget -O model.zip "https://www.dropbox.com/s/d1lfnv6rhc247me/model.ckpt-80000.zip?dl=0"
unzip model.zip -d save/ 
python sample.py --test_file "$1" --result_file "$2"