import tensorflow as tf
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d

def ResBlock(name, inputs, dim):
    output = inputs
    output = tf.nn.leaky_relu(output, 0.2)
    output = lib.ops.conv1d.Conv1D(name+'.1', dim, dim, 5, output)
    output = tf.nn.leaky_relu(output, 0.2)
    output = lib.ops.conv1d.Conv1D(name+'.2', dim, dim, 5, output)
    return inputs + (0.3*output)

def Generator(n_samples, seq_len, layer_dim, output_dim, prev_outputs=None):
    output = make_noise(shape=[n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, seq_len * layer_dim, output)
    output = tf.reshape(output, [-1, layer_dim, seq_len])
    output = ResBlock('Generator.1', output, layer_dim)
    output = ResBlock('Generator.2', output, layer_dim)
    output = ResBlock('Generator.3', output, layer_dim)
    output = ResBlock('Generator.4', output, layer_dim)
    output = ResBlock('Generator.5', output, layer_dim)
    output = lib.ops.conv1d.Conv1D('Generator.Output', layer_dim, output_dim, 1, output)
    output = tf.transpose(output, [0, 2, 1])
    output = softmax(output, output_dim)
    return output

def Discriminator(inputs, seq_len, layer_dim, input_dim, f=1024):
    output = tf.transpose(inputs, [0,2,1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', input_dim, layer_dim, 1, output)
    output = ResBlock('Discriminator.1', output, layer_dim)
    output = ResBlock('Discriminator.2', output, layer_dim)
    output = ResBlock('Discriminator.3', output, layer_dim)
    output = ResBlock('Discriminator.4', output, layer_dim)
    output = ResBlock('Discriminator.5', output, layer_dim)
    # ここまで一緒
    #output = tf.contrib.layers.layer_norm(output)   # ノルムいらないかも（学習不安定になる）
    output = tf.nn.leaky_relu(output, 0.2)
    output = tf.reshape(output, [-1, seq_len * layer_dim]) # seq_len*input_dim
    output = lib.ops.linear.Linear('Discriminator.Output', seq_len * layer_dim, f, output)
    p = tf.divide(2.0*tf.transpose(output), tf.pow(tf.norm(output, axis=1), 2) + 1.0)
    tmp = tf.divide((tf.pow(tf.norm(output, axis=1), 2) - 1.0), (tf.pow(tf.norm(output, axis=1), 2) + 1.0))
    output = tf.concat([p, [tmp]], axis=0)
    output = tf.transpose(output)
    return output

def softmax(logits, num_classes):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes])
        ),
        tf.shape(logits)
    )

def make_noise(shape):
    return tf.random_normal(shape)
