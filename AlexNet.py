import tensorflow as tf
import numpy as np
import os
from datagenerator import ImageDataGenerator
from datetime import datetime
Iterator = tf.data.Iterator
import matplotlib.pyplot as plt

train_file = './train.txt'
val_file = './val.txt'
batch_size = 32
num_classes = 101
num_epoch = 50
learning_rate = 0.001
keep_prob = 0.5


tr_data = ImageDataGenerator(train_file,
                             mode='training',
                             batch_size=batch_size,
                             num_classes=num_classes,
                             shuffle=True)
val_data = ImageDataGenerator(val_file,
                              mode='inference',
                              batch_size=batch_size,
                              num_classes=num_classes,
                              shuffle=False)

# create an reinitializable iterator given the dataset structure
# iterator = Iterator.from_structure(tr_data.data.output_types,
#                                    tr_data.data.output_shapes)
# next_batch = iterator.get_next()

tr_iterator = tr_data.iterator
val_iterator = val_data.iterator
training_init_op = tr_iterator.get_next()
validation_init_op = val_iterator.get_next()
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

def conv(input, out_channels, filter_size, stride, padding, normalization=False, name=None):

    if name==None:
        name = "Conv"
    in_channels = input.get_shape().as_list()[-1]
    with tf.name_scope(name):
        filter = tf.Variable(tf.random_normal(shape=[filter_size[0], filter_size[1], in_channels, out_channels], mean=0, stddev=0.01))
        conv = tf.nn.conv2d(input, filter, [1, stride, stride, 1], padding, name=name)
        if name in ['Conv1', 'Conv3']:
            b = tf.Variable(tf.zeros(shape=(out_channels,)))
        else:
            b = tf.Variable(tf.ones(shape=(out_channels,)))
        activation = tf.nn.relu(tf.nn.bias_add(conv, b), name=name)
        if normalization:
            activation = tf.nn.local_response_normalization(activation, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name=name)

        return activation

def pool(input, filter_size, stride, padding, name=None):

    if name==None:
        name = "Max_pooling"

    with tf.name_scope(name):
        pool = tf.nn.max_pool(input, [1, filter_size[0], filter_size[1], 1], [1, stride, stride, 1], padding, name=name)
        return pool

def fc(input, output_size, dropout=None, name=None):

    if name==None:
        name="Fc"

    with tf.name_scope(name):
        input_shape = input.get_shape().as_list()
        if len(input_shape) == 4:
            fc_weights = tf.Variable(
                tf.random_normal([input_shape[1] * input_shape[2] * input_shape[3], output_size], dtype=tf.float32,
                                 mean=0, stddev=0.01),
                name='weights')
            input = tf.reshape(input, [-1, fc_weights.get_shape().as_list()[0]])
        else:
            fc_weights = tf.Variable(tf.random_normal([input_shape[-1], output_size], dtype=tf.float32, mean=0, stddev=0.01),
                                     name='weights')

        fc_biases = tf.Variable(tf.ones(shape=[output_size], dtype=tf.float32), name='biases')
        fc = tf.nn.relu(tf.nn.xw_plus_b(input, fc_weights, fc_biases), name=name)
        if dropout != None:
            fc = tf.nn.dropout(fc, dropout)
        return fc

def forward_propagation(x, keep_prob):

    # #conv1
    # conv = tf.nn.conv2d(x, Parameters['conv1W'], [1, 4, 4, 1], padding='VALID')
    # conv1 = tf.nn.relu(tf.reshape(tf.nn.bias_add(conv, Parameters['conv1b']), tf.shape(conv)), name='conv1')
    # norm1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name='norm1')
    # pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    # #conv2
    # conv = tf.nn.conv2d(pool1, Parameters['conv2W'], [1, 1, 1, 1], padding='SAME')
    # conv2 = tf.nn.relu(tf.reshape(tf.nn.bias_add(conv, Parameters['conv2b']), tf.shape(conv)), name='conv2')
    # norm2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name='norm2')
    # pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    # #conv3
    # conv = tf.nn.conv2d(pool2, Parameters['conv3W'], [1, 1, 1, 1], padding='SAME')
    # conv3 = tf.nn.relu(tf.reshape(tf.nn.bias_add(conv, Parameters['conv3b']), tf.shape(conv)), name='conv3')
    # #conv4
    # conv = tf.nn.conv2d(conv3, Parameters['conv4W'], [1, 1, 1, 1], padding='SAME')
    # conv4 = tf.nn.relu(tf.reshape(tf.nn.bias_add(conv, Parameters['conv4b']), tf.shape(conv)), name='conv4')
    # #conv5
    # conv = tf.nn.conv2d(conv4, Parameters['conv5W'], [1, 1, 1, 1], padding='SAME')
    # conv5 = tf.nn.relu(tf.reshape(tf.nn.bias_add(conv, Parameters['conv5b']), tf.shape(conv)), name='conv5')
    # pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    # #flatten
    # flattened = tf.reshape(pool5, [-1, 9216])
    # #fc6
    # fc6 = tf.nn.relu(tf.nn.xw_plus_b(flattened, Parameters['fc6W'], Parameters['fc6b']), name='fc6')
    # dropout6 = tf.nn.dropout(fc6, keep_prob)
    # #fc7
    # fc7 = tf.nn.relu(tf.nn.xw_plus_b(dropout6, Parameters['fc7W'], Parameters['fc7b']), name='fc7')
    # dropout7 = tf.nn.dropout(fc7, keep_prob)
    # #fc8
    # fc8 = tf.nn.relu(tf.nn.xw_plus_b(dropout7, Parameters['fc8W'], Parameters['fc8b']), name='fc8')
    # # softmax = tf.nn.softmax(fc8, name='Softmax')

    #Conv1
    Conv1 = conv(x, 96, [11, 11], 4, padding='VALID', normalization=True, name='Conv1')
    #Pool1
    Pool1 = pool(Conv1, [3, 3], 2, padding='VALID', name='Max_pooling1')
    #Conv2
    Conv2 = conv(Pool1, 256, [5, 5], 1, padding='SAME', normalization=True, name='Conv2')
    #Pool2
    Pool2 = pool(Conv2, [3, 3], 2, padding='VALID', name='Max_pooling2')
    #Conv3
    Conv3 = conv(Pool2, 384, [3, 3], 1, padding='SAME', name='Conv3')
    #Conv4
    Conv4 = conv(Conv3, 384, [3, 3], 1, padding='SAME', name='Conv4')
    #Conv5
    Conv5 = conv(Conv4, 256, [3, 3], 1, padding='SAME', name='Conv5')
    #Pool5
    Pool5 = pool(Conv5, [3, 3], 2, padding='VALID', name='Max_pooling5')
    #Fc6
    Fc6 = fc(Pool5, 4096, dropout=keep_prob, name='Fc6')
    #Fc7
    Fc7 = fc(Fc6, 4096, dropout=keep_prob, name='Fc7')
    #Fc8
    Fc8 = fc(Fc7, num_classes, name='Fc8')

    return Fc8


def compute_cost(Y_hat, Y):

    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_hat, labels=Y))


x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
# parameters = initialize_parameters(num_classes)
y_hat = forward_propagation(x, keep_prob)
cost = compute_cost(y_hat, y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


costs = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(num_epoch):

        print("Epoch number: {}".format(epoch + 1))

        # Initialize iterator with the training dataset
        # sess.run(training_init_op)
        sess.run(tr_iterator.initializer)

        for step in range(train_batches_per_epoch):
            # get next batch of data
            img_batch, label_batch = sess.run(training_init_op)

            # And run the training op
            _, temp_cost = sess.run([optimizer, cost], feed_dict={x: img_batch,
                                          y: label_batch})

            costs.append(temp_cost)

            print('Cost after epoch %i: %f' % (epoch+1, temp_cost))


    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()











