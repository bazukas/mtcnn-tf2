import tensorflow as tf
from tensorflow.keras import Model, Input, layers


def PNet(weights=None):
    """ Proposal Network, receives an image and outputs
    bbox offset regressions and confidence scores for each sliding
    window of 12x12
    """
    # img_in = Input(shape=(None, None, 3))

    # # permute needed because of imported weights
    # x = layers.Permute((2, 1, 3), name='permute')(img_in)
    # x = layers.Conv2D(10, 3, 1, name='conv1')(x)
    # x = layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
    # x = layers.MaxPool2D(2, 2, padding='same', name='pool1')(x)

    # x = layers.Conv2D(16, 3, 1, name='conv2')(x)
    # x = layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)

    # x = layers.Conv2D(32, 3, 1, name='conv3')(x)
    # x = layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # # permute needed because of imported weights
    # a = layers.Conv2D(2, 1, 1, name='conv4_1')(x)
    # a = layers.Softmax()(a)
    # # a = tf.transpose(a, (0, 2, 1, 3))
    # a = layers.Permute((2, None, 2))(a)
    # b = layers.Conv2D(4, 1, 1, name='conv4_2')(x)
    # # b = tf.transpose(b, (0, 2, 1, 3))
    # b = layers.Reshape((-1, -1,4))(b)

    # model = Model(inputs=[img_in], outputs=[a, b])

    # if weights is not None:
    #     model.load_weights(weights)
    model = tf.keras.models.load_model(weights)
    return model


def RNet(weights=None):
    """ Refine Network, receives image crops from PNet and outputs
    further offset refinements and confidence scores to filter out
    the predictions
    """
    # img_in = Input(shape=(24, 24, 3))
    # # permute needed because of imported weights
    # x = layers.Permute((2, 1, 3), name='permute')(img_in)
    # x = layers.Conv2D(28, 3, 1, name='conv1')(x)
    # x = layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
    # x = layers.MaxPool2D(3, 2, padding='same', name='pool1')(x)

    # x = layers.Conv2D(48, 3, 1, name='conv2')(x)
    # x = layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)
    # x = layers.MaxPool2D(3, 2, name='pool2')(x)

    # x = layers.Conv2D(64, 2, 1, name='conv3')(x)
    # x = layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # x = layers.Flatten(name='flatten')(x)
    # x = layers.Dense(128, name='dense4')(x)
    # x = layers.PReLU(name='prelu4')(x)

    # a = layers.Dense(2, name='dense5_1')(x)
    # a = layers.Softmax()(a)
    # b = layers.Dense(4, name='dense5_2')(x)

    # model = Model(inputs=[img_in], outputs=[a, b])
    # if weights is not None:
    #     model.load_weights(weights)
    model = tf.keras.models.load_model(weights)
    return model


def ONet(weights=None):
    """ Output Network, receives image crops from RNet and outputs
    final offset regressions, facial landmark positions and confidence scores
    """
    # img_in = Input(shape=(48, 48, 3))
    # # permute needed because of imported weights
    # x = layers.Permute((2, 1, 3), name='permute')(img_in)
    # x = layers.Conv2D(32, 3, 1, name='conv1')(x)
    # x = layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
    # x = layers.MaxPool2D(3, 2, padding='same', name='pool1')(x)

    # x = layers.Conv2D(64, 3, 1, name='conv2')(x)
    # x = layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)
    # x = layers.MaxPool2D(3, 2, name='pool2')(x)

    # x = layers.Conv2D(64, 3, 1, name='conv3')(x)
    # x = layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)
    # x = layers.MaxPool2D(2, 2, padding='same', name='pool3')(x)

    # x = layers.Conv2D(128, 2, 1, name='conv4')(x)
    # x = layers.PReLU(shared_axes=[1, 2], name='prelu4')(x)

    # x = layers.Flatten(name='flatten')(x)
    # x = layers.Dense(256, name='dense5')(x)
    # x = layers.PReLU(name='prelu5')(x)

    # a = layers.Dense(2, name='dense6_1')(x)
    # a = layers.Softmax()(a)
    # b = layers.Dense(4, name='dense6_2')(x)
    # c = layers.Dense(10, name='dense6_3')(x)

    # model = Model(inputs=[img_in], outputs=[a, b, c])
    # if weights is not None:
    #     model.load_weights(weights)
    model = tf.keras.models.load_model(weights)
    return model

# # a = PNet()
# # a.summary()
# from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, PReLU, Flatten, Softmax
# from tensorflow.keras.models import Model

# import numpy as np

# def PNet(url):
#     # if input_shape is None:

#     p_inp = Input(shape=(None, None, 3))

#     p_layer = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding="valid")(p_inp)
#     p_layer = PReLU(shared_axes=[1, 2])(p_layer)
#     p_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(p_layer)

#     p_layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid")(p_layer)
#     p_layer = PReLU(shared_axes=[1, 2])(p_layer)

#     p_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(p_layer)
#     p_layer = PReLU(shared_axes=[1, 2])(p_layer)

#     p_layer_out1 = Conv2D(2, kernel_size=(1, 1), strides=(1, 1))(p_layer)
#     p_layer_out1 = Softmax(axis=3)(p_layer_out1)

#     p_layer_out2 = Conv2D(4, kernel_size=(1, 1), strides=(1, 1))(p_layer)

#     p_net = Model(p_inp, [p_layer_out1, p_layer_out2])
#     weights = np.load(url, allow_pickle=True).tolist()
#     p_net.set_weights(weights['pnet'])
#     return p_net

# def RNet(url):
#     # if input_shape is None:

#     r_inp = Input(shape=(24, 24, 3))

#     r_layer = Conv2D(28, kernel_size=(3, 3), strides=(1, 1), padding="valid")(r_inp)
#     r_layer = PReLU(shared_axes=[1, 2])(r_layer)
#     r_layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(r_layer)

#     r_layer = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding="valid")(r_layer)
#     r_layer = PReLU(shared_axes=[1, 2])(r_layer)
#     r_layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(r_layer)

#     r_layer = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding="valid")(r_layer)
#     r_layer = PReLU(shared_axes=[1, 2])(r_layer)
#     r_layer = Flatten()(r_layer)
#     r_layer = Dense(128)(r_layer)
#     r_layer = PReLU()(r_layer)

#     r_layer_out1 = Dense(2)(r_layer)
#     r_layer_out1 = Softmax(axis=1)(r_layer_out1)

#     r_layer_out2 = Dense(4)(r_layer)

#     r_net = Model(r_inp, [r_layer_out1, r_layer_out2])
#     weights = np.load(url, allow_pickle=True).tolist()
#     r_net.set_weights(weights['rnet'])
#     return r_net

# def ONet(url):
#     # if input_shape is None:
#     # input_shape = 

#     o_inp = Input(shape=(48, 48, 3))
#     o_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(o_inp)
#     o_layer = PReLU(shared_axes=[1, 2])(o_layer)
#     o_layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(o_layer)

#     o_layer = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid")(o_layer)
#     o_layer = PReLU(shared_axes=[1, 2])(o_layer)
#     o_layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(o_layer)

#     o_layer = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid")(o_layer)
#     o_layer = PReLU(shared_axes=[1, 2])(o_layer)
#     o_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(o_layer)

#     o_layer = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), padding="valid")(o_layer)
#     o_layer = PReLU(shared_axes=[1, 2])(o_layer)

#     o_layer = Flatten()(o_layer)
#     o_layer = Dense(256)(o_layer)
#     o_layer = PReLU()(o_layer)

#     o_layer_out1 = Dense(2)(o_layer)
#     o_layer_out1 = Softmax(axis=1)(o_layer_out1)
#     o_layer_out2 = Dense(4)(o_layer)
#     o_layer_out3 = Dense(10)(o_layer)

#     o_net = Model(o_inp, [o_layer_out1, o_layer_out2, o_layer_out3])
#     weights = np.load(url, allow_pickle=True).tolist()
#     o_net.set_weights(weights['onet'])
#     return o_net

# weights = np.load(weights_file, allow_pickle=True).tolist()

# r_net = RNet()
# o_net = ONet()

# p_net.set_weights(weights['pnet'])
# r_net.set_weights(weights['rnet'])
# o_net.set_weights(weights['onet'])

# p_net.summary()