from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import re
import argparse
from six.moves import cPickle

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from model import DCGAN, WGAN, WGAN_v2
from utils import pp, visualize, to_json, show_all_variables

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='WGAN_v2',
                        help='Specify the model to use (DCGAN, WGAN (weight clipping), WGAN_v2 (gradient penalty)) [WGAN_v2]')
    parser.add_argument('--epoch', type=int, default=600, help='Epoch to train [600]')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate of for adam [0.0002]')
    parser.add_argument('--beta1', type=float, default=0.5, help='Momentum term of adam [0.5]')
    parser.add_argument('--clipping_value', type=float, default=0.01, help='Weight clipping value of WGAN [0.01]')
    parser.add_argument('--scale', type=float, default=10.0, help='Gradient penalty scale of WGAN_v2 [10.0]')
    parser.add_argument('--train_size', type=int, default=np.inf, help='The size of train images [np.inf]')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch images [64]')
    parser.add_argument('--y_dim', type=int, default=4800, help='The size of text embedding vector [4800]')
    parser.add_argument('--t_dim', type=int, default=256, help='The size of reduced text embedding vector [256]')
    parser.add_argument('--z_dim', type=int, default=100, help='The size of random noise [100]')
    parser.add_argument('--input_height', type=int, default=64, help='The size of image to use (may be center cropped). [64]')
    parser.add_argument('--input_width', type=int, default=None, help='The size of image to use (may be center cropped). If None, same value as input_height [None]')
    parser.add_argument('--output_height', type=int, default=64, help='The size of the output images to produce [64]')
    parser.add_argument('--output_width', type=int, default=None, help='The size of the output images to produce. If None, same value as output_height [None]')
    parser.add_argument('--dataset', type=str, default='faces', help='The name of dataset [faces]')
    parser.add_argument('--input_fname_pattern', type=str, default='*.jpg', help='Glob pattern of filename of input images [*.jpg]')
    parser.add_argument('--save_dir', type=str, default='save', help='Directory name to save the checkpoints and configurations [save]')
    parser.add_argument('--temp_samples_dir', type=str, default='temp_samples', help='Directory name to save the temp image samples during training [temp_samples]')
    parser.add_argument('--tag_filename', type=str, default='vec_hair_eyes.pkl', help='File name of the embedded tag vectors [vec_hair_eyes.pkl]')
    parser.add_argument('--tag_filename_sp', type=str, default='blonde_hair_blue_eyes.pkl', help='File name of the special embedded tag vectors [blonde_hair_blue_eyes.pkl]')
    parser.add_argument('--crop', type=bool, default=False, help='True for cropping images, False for not [False]')
    parser.add_argument('--visualize', type=bool, default=False, help='True for visualizing images, False for nothing [False]')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""Continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : Configuration;
                            'checkpoint'        : Paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'XXGAN.model-*'     : File(s) with model definition (created by tf)""")
    args = parser.parse_args()
    train(args)

def train(args):
    if args.input_width is None:
        args.input_width = args.input_height
    if args.output_width is None:
        args.output_width = args.output_height

    args.save_dir = args.save_dir + '_' + args.model
    args.temp_samples_dir = args.temp_samples_dir + '_' + args.model
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.temp_samples_dir):
        os.makedirs(args.temp_samples_dir)

    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")), "config.pkl file does not exist in path %s" % args.init_from
        # get ckpt
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        need_be_same=['y_dim','t_dim','z_dim','input_height','input_width','output_height','output_width']
        for checkme in need_be_same:
            assert vars(saved_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme
    else:
        with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
            cPickle.dump(args, f)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=False

    with tf.Session(config=run_config) as sess:
        if args.init_from is not None:
            init_from = args.init_from
            args = saved_args
            args.init_from = init_from
        if args.model == 'DCGAN':
            gan = DCGAN(
                sess,
                args.model,
                input_width=args.input_width,
                input_height=args.input_height,
                output_width=args.output_width,
                output_height=args.output_height,
                batch_size=args.batch_size,
                sample_num=args.batch_size,
                y_dim=args.y_dim,
                t_dim=args.t_dim,
                z_dim=args.z_dim,
                dataset_name=args.dataset,
                input_fname_pattern=args.input_fname_pattern,
                crop=args.crop,
                save_dir=args.save_dir,
                temp_samples_dir=args.temp_samples_dir,
                tag_filename=args.tag_filename,
                tag_filename_sp=args.tag_filename_sp)
        elif args.model == 'WGAN':
            gan = WGAN(
                sess,
                args.model,
                input_width=args.input_width,
                input_height=args.input_height,
                output_width=args.output_width,
                output_height=args.output_height,
                batch_size=args.batch_size,
                sample_num=args.batch_size,
                y_dim=args.y_dim,
                t_dim=args.t_dim,
                z_dim=args.z_dim,
                dataset_name=args.dataset,
                input_fname_pattern=args.input_fname_pattern,
                crop=args.crop,
                save_dir=args.save_dir,
                temp_samples_dir=args.temp_samples_dir,
                tag_filename=args.tag_filename,
                tag_filename_sp=args.tag_filename_sp,
                clipping_value = args.clipping_value)
        elif args.model == 'WGAN_v2':
            gan = WGAN_v2(
                sess,
                args.model,
                input_width=args.input_width,
                input_height=args.input_height,
                output_width=args.output_width,
                output_height=args.output_height,
                batch_size=args.batch_size,
                sample_num=args.batch_size,
                y_dim=args.y_dim,
                t_dim=args.t_dim,
                z_dim=args.z_dim,
                dataset_name=args.dataset,
                input_fname_pattern=args.input_fname_pattern,
                crop=args.crop,
                save_dir=args.save_dir,
                temp_samples_dir=args.temp_samples_dir,
                tag_filename=args.tag_filename,
                tag_filename_sp=args.tag_filename_sp,
                scale = args.scale)

        gan.build_model()

        if args.init_from is not None:
            gan.load(args.init_from)

        show_all_variables()
        gan.train(args)

        # Below is codes for visualization
        OPTION = 1
        # visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    main()
