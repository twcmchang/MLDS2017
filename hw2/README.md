# Quick start
```
bash run.sh MLDS_hw2_data/testing_id.txt MLDS_hw2_data/testing_data/feat/
```

# (1) Train
Put training data into MLDS_hw2_data/
```
python train.py
```
### Default setting
|**Network parameter**| **```n_lstm_step```** | **```n_video_step```** | **```n_caption_step```** | **```dim_image```** | **```dim_hidden```** |
|:-------:|:----:|:----:|:----:|:----:|:----:|
|Default  |  80  |  80  |  20  | 4096 | 1000 |
|**Training parameter** | **```n_epoch```** | **```batch_size```** | **```learning_rate```** | **```grad_clip```** ||
|Default | 1000 |  50  | 0.001 |  10  ||

The model and checkpoint will be stored in save/.

### Schedule sampling (default: 0.0)
Enter an initial sampling probability as follow.
```
python train.py --schedule_sampling 0.01
```
Sampling probability is designed to increase (1+N) times after 50*N epochs.

### Attention-based model (default: 0)
Turn on Attention as follow.
```
python train.py --attention 1
```

# (2) Test
```
python test.py
```
|**Argument**| **```testing_file```** | **```testing_path```** | **```result_file```** | **```init_from```** | 
|:-------:|:----:|:----:|:----:|:----:|
|Default  |MLDS_hw2_data/testing_id.txt|MLDS_hw2_data/testing_data/feat/|output.json|save/| 

If would like to specify testing_file and testing_path,
```
python test.py --testing_file "your_testing_id.txt" --testing_path "your_testing_feat_path"
```

# (3) Evaluate
```
python eval.py --result_file "your_output_json" --test_label_json "your_answer_json"
```
|**Argument**| **```test_label_json```** | **```result_file```** | 
|:-------:|:----:|:----:|
|Default  |MLDS_hw2_data/testing_public_label.json| output.json|

Current best average BLEU score: 0.274922 after 1000 epochs

### Requirement
- Download [dataset][dataset] and unzip it into hw2/
- Tensorflow r1.0

### References
https://github.com/chenxinpeng/S2VT

https://github.com/yunjey/show-attend-and-tell

[Homework slides][slide]

[slide]: https://docs.google.com/presentation/d/1OtD_BD6_Ljvr3aqLjHnnNX_h55BirD3cxhExq9wySmI/edit#slide=id.g1f124951be_0_36
[dataset]: http://speech.ee.ntu.edu.tw/~yangchiyi/MLDS_hw2/MLDS_hw2_data.tar.gz
