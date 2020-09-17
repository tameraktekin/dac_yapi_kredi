from keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout, Concatenate
from keras.applications import vgg16
from keras.models import Model, Sequential


def build_vgg(input_shape):
    vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                                                 input_shape=input_shape)

    output = vgg.layers[-1].output
    output = Flatten()(output)

    vgg_model = Model(vgg.input, output)
    vgg_model.trainable = False

    for layer in vgg_model.layers:
        layer.trainable = False

    vgg_model.summary()

    vgg_model.trainable = True

    set_trainable = False
    for layer in vgg_model.layers:
        if layer.name in ['block5_conv1', 'block4_conv1']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    return vgg_model


def build_model(input_shape):
    x1 = build_vgg(input_shape)
    x2 = build_vgg(input_shape)

    for layer in x1.layers:
        layer.name = layer.name + str('_1')
    for layer in x2.layers:
        layer.name = layer.name + str('_2')

    merged_model = Concatenate()([x1.output, x2.output])
    merged_model = Dense(units=1024)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    merged_model = Activation('relu')(merged_model)
    merged_model = Dropout(0.5)(merged_model)
    merged_model = Dense(units=1024)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    merged_model = Activation('relu')(merged_model)
    merged_model = Dropout(0.5)(merged_model)
    merged_model = Dense(units=1024)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    merged_model = Activation('relu')(merged_model)
    merged_model = Dropout(0.5)(merged_model)
    merged_model = Dense(units=2, activation='softmax')(merged_model)
    model = Model([x1.input, x2.input], merged_model)

    """
    model = Sequential()
    model.add(vgg_model)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=2, activation="softmax"))
    """

    return model
