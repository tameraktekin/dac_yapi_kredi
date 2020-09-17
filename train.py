from utils import *
import numpy as np
from model import *
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.utils import to_categorical

train_dir = "./train (1)/train/"

user_combs = list(organize_data(train_dir))

labels = []
for item in user_combs:
    if item[0][0:3] == item[1][0:3]:
        labels.append(1)
    else:
        labels.append(0)

val_split = 0.2
batch_size = 16
nb_epochs = 50
learning_rate = 0.001
input_shape = (128, 128, 3)
IMG_DIM = input_shape

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1,
                              patience=5, min_lr=0.00001)
callbacks = [reduce_lr]

opt = Adam(learning_rate)
# Shuffle files and labels with the same order.
train_file_list = np.array(user_combs)
labels_list = np.array(labels)

indices = np.arange(train_file_list.shape[0])

np.random.shuffle(indices)

train_file_list = train_file_list[indices]
labels_list = labels_list[indices]

train_length = len(train_file_list)
val_num = int(train_length * val_split)

val_file_list = train_file_list[train_length - val_num: train_length]
val_label_list = labels_list[train_length - val_num: train_length]

val_label_list = to_categorical(val_label_list, 2)

train_file_list = train_file_list[0:train_length - val_num]
labels_list = labels_list[0:train_length - val_num]

labels_list = to_categorical(labels_list, 2)

steps_per_epoch = np.ceil(len(train_file_list)/batch_size)
validation_steps = np.ceil(len(val_file_list)/batch_size)


def load_images(files):
    global IMG_DIM
    global train_dir

    imgs = []

    for file in files:
        img0 = load_img(train_dir + "NFI-" + file[0], target_size=IMG_DIM)
        img0 = img_to_array(img0)
        img0 = img0 / 255

        img1 = load_img(train_dir + "NFI-" + file[1], target_size=IMG_DIM)
        img1 = img_to_array(img1)
        img1 = img1 / 255

        imgs.append([img0, img1])

    return np.array(imgs)


def image_loader(files, labels, batch_size):

    L = len(files)

    # this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = load_images(files[batch_start:limit])
            Y_a = labels[batch_start:limit]

            yield ([X[:,0,:,:,:], X[:,1,:,:,:]],Y_a) # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size


def val_load_images(files):
    global IMG_DIM
    global train_dir

    imgs = []

    for file in files:
        img0 = load_img(train_dir + "NFI-" + file[0], target_size=IMG_DIM)
        img0 = img_to_array(img0)
        img0 = img0 / 255

        img1 = load_img(train_dir + "NFI-" + file[1], target_size=IMG_DIM)
        img1 = img_to_array(img1)
        img1 = img1 / 255

        imgs.append([img0, img1])

    return np.array(imgs)


def val_image_loader(files, labels, batch_size):

    L = len(files)

    # this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = val_load_images(files[batch_start:limit])
            Y_a = labels[batch_start:limit]

            yield ([X[:,0,:,:,:], X[:,1,:,:,:]],[Y_a]) # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size


model = build_model(input_shape)
model.compile(loss='categorical_crossentropy', optimizer=opt)

model.summary()

history = model.fit_generator(generator=image_loader(train_file_list, labels_list, batch_size),
                              steps_per_epoch=steps_per_epoch, epochs=nb_epochs,
                              validation_data=val_image_loader(val_file_list, val_label_list, batch_size),
                              validation_steps=validation_steps, verbose=1, callbacks=callbacks)

model.save("model.h5")
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("accuracy.png")

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("loss.png")

