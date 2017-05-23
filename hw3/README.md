# HW3
Generative Adversarial Networks
- [Slide][slide] & [Dataset][dataset]
- Deadline: 5/25(Thu.) 23:59:59 (UTC+8)

[slide]: https://docs.google.com/presentation/d/1Ea4ywtR5jwiGs-LLkKaaKazxZA37l88vBpjRg7meTB8/edit#slide=id.p
[dataset]: https://drive.google.com/open?id=0BwJmB7alR-AvMHEtczZZN0EtdzQ

## Usage

### Training

(Default) To train a improved WGAN model with image folder 'faces' and text vectors 'vec_hair_eyes.pkl' for 600 epochs:

	$ python train.py

Some samples of generated images will be stored in temp_samples_WGAN_v2, and checkpoints will be stored in save_WGAN_v2.

Or you can specify the model by --model (other choices are DCGAN and WGAN), image folder by --dataset, file of text vectors by --tag_filename, and number of epochs by --epoch, for example:

	$ python train.py --model DCGAN --dataset faces --tag_filename vec_hair_eyes_padding.pkl --epoch 200

Some samples of generated images will be stored in temp_samples_DCGAN, and checkpoints will be stored in save_DCGAN.

For further parameters, please refer to args defined in train.py.

### Generation

To generate images from a saved model and a testing text to a specific folder:

	$ python generate.py --testing_text testing_text.txt --sample_dir samples --init_from save_WGAN_v2

## References

- [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
- [text-to-image](https://github.com/paarthneekhara/text-to-image)

