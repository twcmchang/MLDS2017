from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import re
import argparse
from six.moves import cPickle
import skipthoughts
import scipy.misc

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

from model import DCGAN, WGAN, WGAN_v2
from utils import pp, visualize, to_json, show_all_variables, save_images
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_text', type=str, default='testing_text.txt', help='File name of testing texts [testing_text.txt]')
    # parser.add_argument('--skipthoughts_model', type=str, default='skipthoughts_model.pkl', help='File name of skipthoughts word embedding model [skipthoughts_model.pkl]')
    parser.add_argument('--sample_dir', type=str, default='samples', help='Directory name to save the generated image [samples]')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""Continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : Configuration;
                            'checkpoint'        : Paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'XXGAN.model-*'     : File(s) with model definition (created by tf)""")
    args = parser.parse_args()
    generate(args)

def generate(args):
    assert os.path.isfile(os.path.join(args.init_from,"config.pkl")), "config.pkl file does not exist in path %s" % args.init_from
    # open old config and check if models are compatible
    with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    
    # parse testing texts and encode them
    # with open(args.skipthoughts_model, 'rb') as f:
    #     skipthoughts_model = cPickle.load(f)
    skipthoughts_model = skipthoughts.load_model()
    test_dict = {}
    with open(args.testing_text,'r') as f:
        lines = f.readlines()
        for line in lines:
            idx, desc = re.split(',', line)
            if saved_args.y_dim == 9600:
                hair = re.findall('[a-zA-Z]+ hair', desc, flags=0)
                hair = [re.sub(' hair', '', hair[0])]
                vec_hair = skipthoughts.encode(skipthoughts_model, hair, verbose = False)
                eyes = re.findall('[a-zA-Z]+ eyes', desc, flags=0)
                eyes = [re.sub(' eyes', '', eyes[0])]
                vec_eyes = skipthoughts.encode(skipthoughts_model, eyes, verbose = False)
                test_dict[idx] = np.concatenate([vec_hair,vec_eyes], 1)
            else:
                vec = skipthoughts.encode(skipthoughts_model, [desc.strip()], verbose = False)
                test_dict[idx] = vec

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=False

    with tf.Session(config=run_config) as sess:
        if saved_args.model == 'DCGAN':
            gan = DCGAN(
                sess,
                saved_args.model,
                input_width=saved_args.input_width,
                input_height=saved_args.input_height,
                output_width=saved_args.output_width,
                output_height=saved_args.output_height,
                batch_size=saved_args.batch_size,
                sample_num=saved_args.batch_size,
                y_dim=saved_args.y_dim,
                t_dim=saved_args.t_dim,
                z_dim=saved_args.z_dim,
                dataset_name=saved_args.dataset,
                input_fname_pattern=saved_args.input_fname_pattern,
                crop=saved_args.crop,
                save_dir=saved_args.save_dir,
                temp_samples_dir=saved_args.temp_samples_dir,
                tag_filename=saved_args.tag_filename,
                tag_filename_sp=saved_args.tag_filename_sp,
                infer=True)
        elif saved_args.model == 'WGAN':
            gan = WGAN(
                sess,
                saved_args.model,
                input_width=saved_args.input_width,
                input_height=saved_args.input_height,
                output_width=saved_args.output_width,
                output_height=saved_args.output_height,
                batch_size=saved_args.batch_size,
                sample_num=saved_args.batch_size,
                y_dim=saved_args.y_dim,
                t_dim=saved_args.t_dim,
                z_dim=saved_args.z_dim,
                dataset_name=saved_args.dataset,
                input_fname_pattern=saved_args.input_fname_pattern,
                crop=saved_args.crop,
                save_dir=saved_args.save_dir,
                temp_samples_dir=saved_args.temp_samples_dir,
                tag_filename=saved_args.tag_filename,
                tag_filename_sp=saved_args.tag_filename_sp,
                clipping_value = saved_args.clipping_value,
                infer=True)
        elif saved_args.model == 'WGAN_v2':
            gan = WGAN_v2(
                sess,
                saved_args.model,
                input_width=saved_args.input_width,
                input_height=saved_args.input_height,
                output_width=saved_args.output_width,
                output_height=saved_args.output_height,
                batch_size=saved_args.batch_size,
                sample_num=saved_args.batch_size,
                y_dim=saved_args.y_dim,
                t_dim=saved_args.t_dim,
                z_dim=saved_args.z_dim,
                dataset_name=saved_args.dataset,
                input_fname_pattern=saved_args.input_fname_pattern,
                crop=saved_args.crop,
                save_dir=saved_args.save_dir,
                temp_samples_dir=saved_args.temp_samples_dir,
                tag_filename=saved_args.tag_filename,
                tag_filename_sp=saved_args.tag_filename_sp,
                scale = saved_args.scale,
                infer=True)
        gan.build_model()
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        # show_all_variables()
        could_load, checkpoint_counter = gan.load(args.init_from)
        if could_load:
            # args.sample_dir = args.sample_dir + str(checkpoint_counter)
            if not os.path.exists(args.sample_dir):
                os.makedirs(args.sample_dir)
        else:
            print('load fail!!!')

        for idx,vec in test_dict.items():
            sample_z = np.random.uniform(-1, 1, size=(gan.batch_size, saved_args.z_dim))
            sample_y = np.repeat(vec, gan.batch_size, axis=0)
            samples = sess.run(gan.sampler, feed_dict={gan.z: sample_z, gan.y: sample_y})
            print(samples.shape)
            ## Randomly take 5 images for each tag
            for sid in range(5):
                # manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                # manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                # save_images(samples[sid], [1, 1],
                #             os.path.join(args.sample_dir, 'sample_{}_{}.jpg'.format(str(idx), str(sid+1))))
                scipy.misc.imsave(os.path.join(args.sample_dir, 'sample_{}_{}.jpg'.format(str(idx), str(sid+1))), samples[sid])

if __name__ == '__main__':
    main()
