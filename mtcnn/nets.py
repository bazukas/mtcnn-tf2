import tensorflow as tf
from tensorflow.keras import Model, Input, layers


def PNet(weights=None):
    """ Proposal Network, receives an image and outputs
    bbox offset regressions and confidence scores for each sliding
    window of 12x12
    """
    img_in = Input(shape=(None, None, 3))

    # permute needed because of imported weights
    x = layers.Permute((2, 1, 3), name='permute')(img_in)
    x = layers.Conv2D(10, 3, 1, name='conv1')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = layers.MaxPool2D(2, 2, padding='same', name='pool1')(x)

    x = layers.Conv2D(16, 3, 1, name='conv2')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)

    x = layers.Conv2D(32, 3, 1, name='conv3')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # permute needed because of imported weights
    a = layers.Conv2D(2, 1, 1, name='conv4_1')(x)
    a = layers.Softmax()(a)
    a = tf.transpose(a, (0, 2, 1, 3))
    b = layers.Conv2D(4, 1, 1, name='conv4_2')(x)
    b = tf.transpose(b, (0, 2, 1, 3))

    model = Model(inputs=[img_in], outputs=[a, b])
    if weights is not None:
        model.load_weights(weights)
    return model


def RNet(weights=None):
    """ Refine Network, receives image crops from PNet and outputs
    further offset refinements and confidence scores to filter out
    the predictions
    """
    img_in = Input(shape=(24, 24, 3))
    # permute needed because of imported weights
    x = layers.Permute((2, 1, 3), name='permute')(img_in)
    x = layers.Conv2D(28, 3, 1, name='conv1')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = layers.MaxPool2D(3, 2, padding='same', name='pool1')(x)

    x = layers.Conv2D(48, 3, 1, name='conv2')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = layers.MaxPool2D(3, 2, name='pool2')(x)

    x = layers.Conv2D(64, 2, 1, name='conv3')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)

    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(128, name='dense4')(x)
    x = layers.PReLU(name='prelu4')(x)

    a = layers.Dense(2, name='dense5_1')(x)
    a = layers.Softmax()(a)
    b = layers.Dense(4, name='dense5_2')(x)

    model = Model(inputs=[img_in], outputs=[a, b])
    if weights is not None:
        model.load_weights(weights)
    return model


def ONet(weights=None):
    """ Output Network, receives image crops from RNet and outputs
    final offset regressions, facial landmark positions and confidence scores
    """
    img_in = Input(shape=(48, 48, 3))
    # permute needed because of imported weights
    x = layers.Permute((2, 1, 3), name='permute')(img_in)
    x = layers.Conv2D(32, 3, 1, name='conv1')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = layers.MaxPool2D(3, 2, padding='same', name='pool1')(x)

    x = layers.Conv2D(64, 3, 1, name='conv2')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = layers.MaxPool2D(3, 2, name='pool2')(x)

    x = layers.Conv2D(64, 3, 1, name='conv3')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = layers.MaxPool2D(2, 2, padding='same', name='pool3')(x)

    x = layers.Conv2D(128, 2, 1, name='conv4')(x)
    x = layers.PReLU(shared_axes=[1, 2], name='prelu4')(x)

    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(256, name='dense5')(x)
    x = layers.PReLU(name='prelu5')(x)

    a = layers.Dense(2, name='dense6_1')(x)
    a = layers.Softmax()(a)
    b = layers.Dense(4, name='dense6_2')(x)
    c = layers.Dense(10, name='dense6_3')(x)

    model = Model(inputs=[img_in], outputs=[a, b, c])
    if weights is not None:
        model.load_weights(weights)
    return model
