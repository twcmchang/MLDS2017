# HW3
Generative Adversarial Networks
- [Slide][slide] & [Dataset][dataset]
- Deadline: 5/25(Thu.) 23:59:59 (UTC+8)

[slide]: https://docs.google.com/presentation/d/1Ea4ywtR5jwiGs-LLkKaaKazxZA37l88vBpjRg7meTB8/edit#slide=id.p
[dataset]: https://drive.google.com/open?id=0BwJmB7alR-AvMHEtczZZN0EtdzQ

## Usage

To train a model with image folder 'faces' and text vectors 'vec_hair_eyes.p' for 600 epochs:

	$ python main.py --train

Or you can specify the image folder by --dataset, file of text vectors by --tag_filename, and number of epochs by --epoch, for example:

	$ python main.py --dataset mnist --tag_filename vec_hair_eyes.p --epoch 200 --train

For further arguments, please refer to flags defined in main.py.

## References

- [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
- [text-to-image](https://github.com/paarthneekhara/text-to-image)

