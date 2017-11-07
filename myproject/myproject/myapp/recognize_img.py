# coding=utf-8
import tensorflow as tf
import os
from PIL import Image


def classifier(image):
    size = 56, 56
    # Parameters
    # Network Parameters
    n_input = size[0] * size[1]  # data input (img shape: 28*28)
    n_classes = 3  # total classes (0-9 digits)

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        # MaxPool2D wrapper (NOTE: ksize = strides, padding = 'SAME')
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    # Create model
    def conv_net(x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, size[0], size[1], 1])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Convolution Layer
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
        # Max Pooling (down-sampling)
        conv3 = maxpool2d(conv3, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # 5x5 conv, 64 inputs, 32 outputs
        'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        # 'wd1': tf.Variable(tf.random_normal([size[0] / 8 * size[1] / 8 * 128, 1024])),
        'wd1': tf.Variable(tf.random_normal([int(size[0] / 8 * size[1] / 8 * 128), 1024])),

        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bc3': tf.Variable(tf.random_normal([128])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

    # image = Image.open(image)
    # image = image.convert('L').resize(size)
    # batch_x = np.fromstring(image.tobytes(), dtype=np.uint8).reshape(1, size[0] * size[1])
    # batch_x = batch_x * (1. / 255) - 0.5

    # 初始化所有的op
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)
    saver.restore(sess, os.getcwd() + "/myproject/myapp/model/model.ckpt")
    result = sess.run(tf.argmax(pred, 1), feed_dict={x: image, keep_prob: 1.})
    sess.close()
    return result


if __name__ == '__main__':
    cwd = os.getcwd()
    img = Image.open('recongnize_img_rs/0/img3.jpg')

    prediction_y = classifier(img)

    print("Predictions:", prediction_y)
    # print("Accuracy:", accuracy)
