# HW2
## Requirement
- Download [dataset][dataset] and unzip it into hw2/
- Tensorflow r1.0

## Train a video caption generator with default setting: 
```
python train.py
```
|**Network parameter**| **```n_lstm_step```** | **```n_video_step```** | **```n_caption_step```** | **```dim_image```** | **```dim_hidden```** |
|:-------:|:----:|:----:|:----:|:----:|:----:|
|Default  |  80  |  80  |  20  | 4096 | 1000 |
|**Training parameter** | **```n_epoch```** | **```batch_size```** | **```learning_rate```** | **```grad_clip```** ||
|Default | 1000 |  50  | .001 |  10  ||

<!---
| Parameter      | Default |
| :------------- | ------: |
| ```n_lstm_step```   | 80 |
| ```n_video_step```  | 80 |
| ```n_caption_step```| 20 |
| ```dim_hidden```    | 1000 |
| ```dim_image```     | 4096 |
| ```n_epoch```       | 1000 |
| ```batch_size```    | 50 |
| ```learning_rate``` | .001 |
| ```grad_clip```     | 10|
--->

After training, model/checkpoint will be stored in a specfic direcotry (default: save/).

## Test your video caption generator model:
```
python test.py
```
'output.json' will be generated

## Evaluate the average BLEU score
```
python eval.py
```

## Resources
https://github.com/chenxinpeng/S2VT

https://github.com/yunjey/show-attend-and-tell

[Homework slides][slide]

## Deadline: 4/27(Thu.) 23:59:59 (UTC+8) 

[slide]: https://docs.google.com/presentation/d/1OtD_BD6_Ljvr3aqLjHnnNX_h55BirD3cxhExq9wySmI/edit#slide=id.g1f124951be_0_36
[dataset]: http://speech.ee.ntu.edu.tw/~yangchiyi/MLDS_hw2/MLDS_hw2_data.tar.gz






