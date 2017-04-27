#!/bin/bash
wget -O model.zip "https://www.dropbox.com/s/pb7zjd7xy9cjyfq/model.zip?dl=0"
unzip model.zip -d save/ 
python3 test.py --testing_file "$1" --testing_path "$2" --result_file "$3"
