import os, sys

sys.path.append(os.getcwd())

import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from statistics import mean


import utils
import tflib as lib
import tflib.plot
import model_SpherePassGAN2 as models


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-data', '-i',
                        default='data/train.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line) (default: data/train.txt)')

    parser.add_argument('--output-dir', '-o',
                        required=True,
                        dest='output_dir',
                        help='Output directory. If directory doesn\'t exist it will be created.')

    parser.add_argument('--save-every', '-s',
                        type=int,
                        default=5000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations (default: 5000)')

    parser.add_argument('--iters', '-n',
                        type=int,
                        default=199000,
                        dest='iters',
                        help='The number of training iterations (default: 199000)')

    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')

    parser.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length (default: 10)')

    parser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator and discriminator (default: 128)')

    parser.add_argument('--critic-iters', '-c',
                        type=int,
                        default=10,
                        dest='critic_iters',
                        help='The number of discriminator weight updates per generator update (default: 10)')

    parser.add_argument('--lambda', '-p',
                        type=int,
                        default=10,
                        dest='lamb',
                        help='The gradient penalty lambda hyperparameter (default: 10)')
    parser.add_argument('--feature-dim', '-f',
                        type=int,
                        default=1024,
                        dest='feature_dim',
                        help='The hypersphere dimension (default: 1024)')
    return parser.parse_args()

args = parse_args()

lines, charmap, inv_charmap = utils.load_dataset(
    path=args.training_data,
    max_length=args.seq_length
)

def get_reference_point(coord=None):
    if coord == None:
        ref_p_np = np.zeros((1, args.feature_dim + 1)).astype(np.float32)
        ref_p_np[0, args.feature_dim] = 1.0
        return tf.constant(ref_p_np)
    else:
        return coord

def _dist_sphere(a, b):
    return tf.acos(tf.matmul(a, tf.transpose(b)))

def dist_weight_mode(r, weight_mode):
    if weight_mode == 'normalization':
        decayed_dist = ((1.0 / 3.0) * np.pi) ** r   #3 is decay_ratio
    elif weight_mode == 'half':
        decayed_dist = (np.pi) ** r
    else:
        decayed_dist = 1.0
    return decayed_dist

def eval_moments(y_pred, moments, weight_mode):
    ref_p = get_reference_point()
    d = 0.0
    for r in range(1, moments + 1):
        d = d + tf.pow(_dist_sphere(y_pred, ref_p), float(r)) / dist_weight_mode(r, weight_mode)
    return d

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

if not os.path.isdir(os.path.join(args.output_dir, 'checkpoints')):
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'))

if not os.path.isdir(os.path.join(args.output_dir, 'samples')):
    os.makedirs(os.path.join(args.output_dir, 'samples'))

# pickle to avoid encoding errors with json
with open(os.path.join(args.output_dir, 'charmap.pickle'), 'wb') as f:
    pickle.dump(charmap, f)

with open(os.path.join(args.output_dir, 'inv_charmap.pickle'), 'wb') as f:
    pickle.dump(inv_charmap, f)

real_inputs_discrete = tf.placeholder(tf.int32, shape=[args.batch_size, args.seq_length])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims - 1)

# Π(D(z)), Π(D(G(z)))
disc_real = models.Discriminator(real_inputs, args.seq_length, args.layer_dim, len(charmap), args.feature_dim)
disc_fake = models.Discriminator(fake_inputs, args.seq_length, args.layer_dim, len(charmap), args.feature_dim)

# 損失関数定義
moments = 1  # 3 is better in cifara10 dataset
weight_mode = None
disc_real_d = eval_moments(disc_real, moments, weight_mode)
disc_fake_d = eval_moments(disc_fake, moments, weight_mode)
disc_cost = tf.reduce_mean(disc_fake_d) - tf.reduce_mean(disc_real_d)
gen_cost = -tf.reduce_mean(disc_fake_d)

# WGAN lipschitz-penalty
alpha = tf.random_uniform(
    shape=[args.batch_size,1,1],
    minval=0.,
    maxval=1.
)

differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha*differences)
d = eval_moments(models.Discriminator(interpolates, args.seq_length, args.layer_dim, len(charmap), args.feature_dim), moments, weight_mode)
gradients = tf.gradients(d, [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
gradient_penalty = tf.reduce_mean(slopes**2)
#gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
disc_cost += args.lamb * gradient_penalty


gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

# 活性化関数定義
gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)


# Dataset iterator
def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines) - args.batch_size + 1, args.batch_size):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i + args.batch_size]],
                dtype='int32'
            )


with tf.Session() as session:
    session.run(tf.global_variables_initializer())


    def generate_samples():
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in range(len(samples)):
            decoded = []
            for j in range(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples


    gen = inf_train_gen()

    for iteration in range(args.iters):
        start_time = time.time()

        # Train generator
        if iteration > 0:
            # _ = session.run(gen_train_op)
            _gen_cost, _ = session.run(
                [gen_cost, gen_train_op]
            )


        # Train critic
        for i in range(args.critic_iters):
            _data = gen.__next__()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_inputs_discrete: _data}
            )


        lib.plot.output_dir = args.output_dir
        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('train disc cost', _disc_cost)
        if iteration > 0:
            lib.plot.plot('train gen cost', _gen_cost)


        if iteration % 100 == 0 and iteration > 0:
            samples = []
            for i in range(10):
                samples.extend(generate_samples())

            with open(os.path.join(args.output_dir, 'samples', 'samples_{}.txt').format(iteration), 'w') as f:
                for s in samples:
                    s = "".join(s)
                    s = (s + "\n").encode('UTF-8')
                    f.buffer.write(s)

        if iteration % args.save_every == 0 and iteration > 0 or iteration == args.iters-1:
            model_saver = tf.train.Saver()
            model_saver.save(session, os.path.join(args.output_dir, 'checkpoints', 'checkpoint_{}.ckpt').format(iteration))

        if iteration % 100 == 0:
            lib.plot.flush()

        lib.plot.tick()
