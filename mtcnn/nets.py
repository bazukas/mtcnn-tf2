import tensorflow as tf
from tensorflow.keras import Model, layers


class PNet(Model):
    def __init__(self):
        super(PNet, self).__init__()
        self.features = [
            layers.Conv2D(10, 3, 1, input_shape=(12, 12, 3), name='conv1'),
            layers.PReLU(shared_axes=[1, 2], name='prelu1'),
            layers.MaxPool2D(2, 2, padding='same', name='pool1'),

            layers.Conv2D(16, 3, 1, name='conv2'),
            layers.PReLU(shared_axes=[1, 2], name='prelu2'),

            layers.Conv2D(32, 3, 1, name='conv3'),
            layers.PReLU(shared_axes=[1, 2], name='prelu3'),
        ]

        self.conv4_1 = layers.Conv2D(2, 1, 1, name='conv4_1')
        self.conv4_2 = layers.Conv2D(4, 1, 1, name='conv4_2')

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None, None, 3),
                                       dtype=tf.float32)])
    def call(self, x):
        for l in self.features:
            x = l(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = layers.Softmax()(a)
        return b, a

    @classmethod
    def load(cls, path):
        m = cls()
        m.build((None, 12, 12, 3))
        m.load_weights(path)
        return m


class RNet(Model):
    def __init__(self):
        super(RNet, self).__init__()
        self.features = [
            layers.Conv2D(28, 3, 1, input_shape=(24, 24, 3), name='conv1'),
            layers.PReLU(shared_axes=[1, 2], name='prelu1'),
            layers.MaxPool2D(3, 2, padding='same', name='pool1'),

            layers.Conv2D(48, 3, 1, name='conv2'),
            layers.PReLU(shared_axes=[1, 2], name='prelu2'),
            layers.MaxPool2D(3, 2, name='pool2'),

            layers.Conv2D(64, 2, 1, name='conv3'),
            layers.PReLU(shared_axes=[1, 2], name='prelu3'),

            layers.Permute((3, 1, 2), name='permute'),
            layers.Flatten(name='flatten'),
            layers.Dense(128, name='dense4'),
            layers.PReLU(name='prelu4'),
        ]

        self.dense5_1 = layers.Dense(2, name='dense5_1')
        self.dense5_2 = layers.Dense(4, name='dense5_2')

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, 24, 24, 3),
                                       dtype=tf.float32)])
    def call(self, x):
        for l in self.features:
            x = l(x)
        a = self.dense5_1(x)
        b = self.dense5_2(x)
        a = layers.Softmax()(a)
        return b, a

    @classmethod
    def load(cls, path):
        m = cls()
        m.build((None, 24, 24, 3))
        m.load_weights(path)
        return m


class ONet(Model):
    def __init__(self):
        super(ONet, self).__init__()
        self.features = [
            layers.Conv2D(32, 3, 1, input_shape=(48, 48, 3), name='conv1'),
            layers.PReLU(shared_axes=[1, 2], name='prelu1'),
            layers.MaxPool2D(3, 2, padding='same', name='pool1'),

            layers.Conv2D(64, 3, 1, name='conv2'),
            layers.PReLU(shared_axes=[1, 2], name='prelu2'),
            layers.MaxPool2D(3, 2, name='pool2'),

            layers.Conv2D(64, 3, 1, name='conv3'),
            layers.PReLU(shared_axes=[1, 2], name='prelu3'),
            layers.MaxPool2D(2, 2, padding='same', name='pool3'),

            layers.Conv2D(128, 2, 1, name='conv4'),
            layers.PReLU(shared_axes=[1, 2], name='prelu4'),

            layers.Permute((3, 1, 2), name='permute'),
            layers.Flatten(name='flatten'),
            layers.Dense(256, name='dense5'),
            layers.Dropout(0.25, name='drop5'),
            layers.PReLU(name='prelu5'),
        ]

        self.dense6_1 = layers.Dense(2, name='dense6_1')
        self.dense6_2 = layers.Dense(4, name='dense6_2')
        self.dense6_3 = layers.Dense(10, name='dense6_3')

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, 48, 48, 3),
                                       dtype=tf.float32)])
    def call(self, x):
        for l in self.features:
            x = l(x)
        a = self.dense6_1(x)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        a = layers.Softmax()(a)
        return c, b, a

    @classmethod
    def load(cls, path):
        m = cls()
        m.build((None, 48, 48, 3))
        m.load_weights(path)
        return m
