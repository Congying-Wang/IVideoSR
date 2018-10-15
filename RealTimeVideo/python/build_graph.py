import os
import time
import glob
import random
import numpy as np
import tensorflow as tf
from subpixel import PS
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class Config(object):
    """Holds model hyperparams and data information.
    """
    batch_size = 32
    n_epochs = 1500
    lr = 0.0001
    regularize_factor = 0.0001
    # Model : 'edsr', 'vdsr', 'resnet' is implemented
    model = 'edsr'
    # Loss : choose between 'l1_loss', 'l2_loss'
    loss = 'l2_loss'


def conv2d(x, input_filters, output_filters, kernel, strides, padding='SAME'):
    with tf.variable_scope('conv'):
        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        bias = tf.Variable(tf.truncated_normal([output_filters], 0.1), name='bias')
        convolved = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding=padding, name='conv')
        conv_result = tf.nn.bias_add(convolved, bias)
        return conv_result


class CNN4SR(object):
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, [None, 72, 72, 3], 'input_image')
        self.y_placeholder = tf.placeholder(tf.float32, [None, 144, 144, 3], 'correct_image')

    def create_feed_dict(self, input_image_batch, correct_image_batch=None):
        feed_dict = dict()
        feed_dict[self.input_placeholder] = input_image_batch
        if correct_image_batch is not None:
            feed_dict[self.y_placeholder] = correct_image_batch
        return feed_dict

    @staticmethod
    def enhanced_residual_cell(input_tensor, scale_factor=1.0,
                               input_filters=3, output_filters=3, kernel=5, stride=1):
        conv1 = conv2d(input_tensor, input_filters, output_filters, kernel, stride)
        elu1 = tf.nn.elu(conv1)
        conv2 = conv2d(elu1, input_filters, output_filters, kernel, stride)
        output = input_tensor + scale_factor * conv2
        return output

    def inference_resnet(self):
        input_img = self.input_placeholder
        conv1 = conv2d(input_img, 3, 6, 5, 1)
        x = tf.nn.elu(conv1)
        conv2 = conv2d(x, 6, 12, 5, 1)
        x = tf.nn.elu(conv2)
        conv3 = conv2d(x, 12, 24, 3, 1)
        x = tf.nn.elu(conv3)
        conv4 = conv2d(x, 24, 48, 3, 1)
        x = tf.nn.elu(conv4)
        conv5 = conv2d(x, 48, 64, 3, 1)
        x = tf.nn.elu(conv5)
        for i in range(32):
            with tf.name_scope('enhanced_residual_net_%d' % i):
                x = self.enhanced_residual_cell(x, 0.1, 64, 64, 3, 1)
        x += conv5
        conv6 = conv2d(x, 64, 48, 3, 1)
        x = tf.nn.elu(conv6 + conv4)
        conv7 = conv2d(x, 48, 24, 3, 1)
        x = tf.nn.elu(conv7 + conv3)
        conv8 = conv2d(x, 24, 12, 3, 1)
        x = tf.nn.elu(conv8 + conv2)
        conv9 = conv2d(x, 12, 12, 3, 1)
        x = tf.nn.elu(conv9)
        conv10 = conv2d(x, 12, 12, 3, 1)
        middle_feature_maps = conv10
        # middle_feature_maps = conv2d(x, 6, 12, 3, 1)
        # pred = tf.nn.tanh(PS(elu_middle, 2, color=True))
        pred = tf.nn.sigmoid(PS(middle_feature_maps, 2, color=True))
        return pred

    def inference_edsr(self):
        input_img = self.input_placeholder
        conv1 = conv2d(input_img, 3, 6, 5, 1)
        x = tf.nn.elu(conv1)
        conv2 = conv2d(x, 6, 12, 5, 1)
        x = tf.nn.elu(conv2)
        conv3 = conv2d(x, 12, 36, 3, 1)
        x = tf.nn.elu(conv3)
        conv4 = conv2d(x, 36, 64, 3, 1)
        x = tf.nn.elu(conv4)
        for i in range(32):
            with tf.name_scope('enhanced_residual_net_%d' % i):
                x = self.enhanced_residual_cell(x, 0.1, 64, 64, 3, 1)
        x += conv4
        conv5 = conv2d(x, 64, 36, 3, 1)
        x = tf.nn.elu(conv5 + conv3)
        conv6 = conv2d(x, 36, 12, 3, 1)
        x = tf.nn.elu(conv6 + conv2)
        conv9 = conv2d(x, 12, 12, 3, 1)
        middle_feature_maps = conv9
        # middle_feature_maps = conv2d(x, 6, 12, 3, 1)
        # pred = tf.nn.tanh(PS(elu_middle, 2, color=True))
        shuffled_img = PS(middle_feature_maps, 2, color=True)
        conv10 = conv2d(shuffled_img, 3, 3, 3, 1)
        pred = tf.nn.sigmoid(conv10)
        return pred

    def inference_vdsr(self):
        input_img = self.input_placeholder
        input_big_img = tf.image.resize_images(input_img, (144, 144), tf.image.ResizeMethod.BICUBIC)
        with tf.name_scope('input_conv'):
            conv1 = conv2d(input_big_img, 3, 9, 3, 1)
            x = tf.nn.elu(conv1)
            conv2 = conv2d(x, 9, 32, 3, 1)
            x = tf.nn.elu(conv2)
            conv3 = conv2d(x, 32, 64, 3, 1)
            x = tf.nn.elu(conv3)
        for i in range(18):
            with tf.name_scope('middle_conv_%d' % i):
                x = tf.nn.elu(conv2d(x, 64, 64, 3, 1))
        with tf.name_scope('output_conv'):
            conv4 = conv2d(x, 64, 32, 3, 1)
            x = tf.nn.elu(conv4)
            conv5 = conv2d(x, 32, 9, 3, 1)
            x = tf.nn.elu(conv5)
            conv6 = conv2d(x, 9, 3, 3, 1)
        with tf.name_scope('residual'):
            pred = input_big_img + conv6
        return pred

    def add_prediction_op(self):
        pred = None
        if self.config.model == 'resnet':
            pred = self.inference_resnet()
        elif self.config.model == 'vdsr':
            pred = self.inference_vdsr()
        elif self.config.model == 'edsr':
            pred = self.inference_edsr()
        else:
            raise ValueError('Config.model should specify a model.')
        return pred

    def add_loss_op(self, pred):
        if self.config.loss == 'l1_loss':
            orig_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(pred - self.y_placeholder), [1, 2, 3]), [0])
        elif self.config.loss == 'l2_loss':
            orig_loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred - self.y_placeholder), [1, 2, 3]), [0])
        else:
            raise ValueError('Config.loss should specify a loss function.')
        var_list = tf.trainable_variables()
        l2_reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in var_list]) * self.config.regularize_factor
        loss = orig_loss + l2_reg_loss
        return loss

    def compute_psnr(self, pred):
        diff = pred - self.y_placeholder
        max_val = 1.0
        mse = tf.reduce_mean(tf.square(diff), [-3, -2, -1])
        psnr = tf.subtract(20 * tf.log(max_val) / tf.log(10.0), np.float32(10 / np.log(10)) * tf.log(mse))
        return psnr

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def predict_on_batch(self, sess, inputs_batch):
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.psnr = self.compute_psnr(self.pred)
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def train_on_batch(self, sess, inputs_batch, correct_image_batch):
        feed = self.create_feed_dict(input_image_batch=inputs_batch,
                                     correct_image_batch=correct_image_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def train(self, sess, train_examples_x, train_examples_y, saver):
        n_minibatches = len(train_examples_x) / self.config.batch_size + 1
        max_psnr = 0
        best_epoch = 0
        for epoch in range(1, 1+self.config.n_epochs):
            print 'Epoch %d started!' % epoch
            print 'Shuffling...'
            data_to_shuffle = zip(train_examples_x, train_examples_y)
            random.Random(0).shuffle(data_to_shuffle)
            train_examples_x, train_examples_y = zip(*data_to_shuffle)
            for i in range(n_minibatches):
                train_x = train_examples_x[i:i+self.config.batch_size]
                train_y = train_examples_y[i:i+self.config.batch_size]
                loss = self.train_on_batch(sess, train_x, train_y)
                print 'Batch %d Training loss = %f' % (i+1, loss)

            eval_train_x = train_examples_x[:100]
            eval_train_y = train_examples_y[:100]
            feed = self.create_feed_dict(eval_train_x, eval_train_y)
            psnr = np.mean(sess.run([self.psnr], feed_dict=feed)[0])
            print 'PSNR on first 100 training set : %f' % psnr

            psnr = self.eval_by_psnr(sess, epoch)
            if psnr > max_psnr:
                print 'New best PSNR = %f!' % psnr
                saver.save(sess, "../data/model/%s/SR_Model_Epoch_%d" % (self.config.model, epoch))
                max_psnr = psnr
                best_epoch = epoch
            else:
                print 'Now PSNR = %f, best PSNR = %f, will not save the current model' % (psnr, max_psnr)
        print 'Best PSNR of model is : %f on epoch %d' % (max_psnr, best_epoch)

    def eval_by_psnr(self, sess, num_epoch):
        eval_files = glob.glob('../data/predict/eval*')
        eval_images_small = []
        eval_images_large = []
        for filepath in eval_files:
            img = Image.open(filepath)
            small_img = img.resize((72, 72), Image.ANTIALIAS)
            orig_img = np.array(img, dtype=np.float32) / 255.0
            img_for_eval = np.array(small_img, dtype=np.float32) / 255.0
            eval_images_small.append(img_for_eval)
            eval_images_large.append(orig_img)

        feed = self.create_feed_dict(eval_images_small, eval_images_large)
        res, psnr = sess.run([self.pred, self.psnr], feed_dict=feed)
        print 'Evaluated PSNR :'
        for i in range(len(eval_files)):
            filename = eval_files[i].split('/')[-1]
            print filename, psnr[i]
            if num_epoch % 10 == 0:
                img = Image.fromarray(np.uint8(res[i] * 255.0))
                img.save(os.path.join('../data/predict_%s/epoch_%d_%s' % (self.config.model, num_epoch, filename)))
        return np.mean(psnr)

    def __init__(self, config):
        self.config = config
        self.build()


def main():
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    config = Config()

    train_examples_x = list()
    train_examples_y = list()
    input_datapath = '../data/train_144/'
    filepaths = glob.glob(input_datapath + '*.bmp')
    for filepath in filepaths:
        img = Image.open(filepath)
        train_examples_y.append(np.array(img, dtype=np.float32) / 255.0)
        train_examples_x.append(np.array(img.resize((72, 72), Image.ANTIALIAS), dtype=np.float32) / 255.0)

    with tf.Graph().as_default() as graph:
        print "Building model...",
        start = time.time()
        model = CNN4SR(config)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        print "took {:.2f} seconds\n".format(time.time() - start)
    graph.finalize()

    with tf.Session(graph=graph) as session:
        session.run(init_op)
        print 80 * "="
        print "TRAINING"
        print 80 * "="
        model.train(session, train_examples_x, train_examples_y, saver)


if __name__ == '__main__':
    main()

