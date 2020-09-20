import argparse
import os

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import videoto3d
from tqdm import tqdm
from  losses import categorical_focal_loss
import keras.backend as K
# import tensorflow as tf
from mertric import fbeta_score

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"]='1'
def mean_pred(y_true, y_pred):
    # f1_list = fbeta_score(y_true, y_pred)
    # print('f1_list',f1_list)
    f1_list = f1_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average=None)
    print('f1_list',f1_list.shape)
    f1 = 0.1 * f1_list[0] + 0.2 * f1_list[1] + 0.3 * f1_list[2]+ 0.4 * f1_list[3]
    return f1

# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy', mean_pred])

def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_accuracy'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))



def loaddata(img_id_dir,img_label_dir, vid3d, nclass, result_dir, color=0):
    files = os.listdir(img_id_dir)
    files.sort()
    # print('video_dir',img_id_dir)
    # print(len(files))

    X = []
    files_ID=[]
    pbar = tqdm(total=len(files))
    for filename in files:
        # print(files[0:3])
        # print('filename',filename)
        X_img=[]
        pbar.update(1)
        path_imgs=img_id_dir+'/'+filename
        file_imgs=os.listdir(path_imgs)
        if len(file_imgs) >=15 :
            for filename_imgs in file_imgs:

                # if filename == '.DS_Store':
                #     continue
                name = os.path.join(path_imgs, filename_imgs)
                # print('name',name)
                #label = get_UCF_classname(filename)
                # if label not in labellist:
                #     if len(labellist) >= nclass:
                #         continue
                #     labellist.append(label)
                # labels.append(label)
                X_img.append(vid3d.loadimg(name, color=color))
            X.append(X_img)
        else:

            files_ID.append(filename)
            # print('files_ID', files_ID)
    pbar.close()

    labels=vid3d.get_img_classname(img_label_dir,files_ID)
    # print('label',labels)
    if color:

        return np.array(X).transpose((0, 2, 3, 4, 1)), labels
    else:
        # print('xshape',np.array(X).shape)
        return np.array(X).transpose((0, 2, 3, 1)), labels

def main():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='UCF101',
                        help='directory where videos are stored')
    parser.add_argument('--img_label_dir', type=str),
    parser.add_argument('--nclass', type=int, default=4)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--color', type=bool, default=0)
    # parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=15)
    args = parser.parse_args()

    img_rows, img_cols, frames = 32, 32, args.depth
    channel = 3 if args.color else 1
    # print('arg.color',args.color)
    fname_npz = 'dataset_{}_{}.npz'.format(
        args.nclass, args.depth)

    vid3d = videoto3d.LoadImg(img_rows, img_cols, frames)
    print('vid3d,',vid3d)
    nb_classes = args.nclass
    if os.path.exists(fname_npz):

        loadeddata = np.load(fname_npz)
        X, Y = loadeddata["X"], loadeddata["Y"]
    else:
        x, y = loaddata(args.videos,args.img_label_dir, vid3d, args.nclass,
                        args.output, args.color)
        X = x.reshape((x.shape[0], img_rows, img_cols, frames, channel))
        Y = np_utils.to_categorical(y, nb_classes)#变成one_hot类型

        X = X.astype('float32')
        np.savez(fname_npz, X=X, Y=Y)
        print('Saved dataset to dataset.npz.')
    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))
    Y_new=np.load(r'label.npy')
    Y_new = np.delete(Y_new, 216, 0)
    Y_new = Y_new[:, 1]
    Y_new = np_utils.to_categorical(Y_new, nb_classes)
    # Define model
    model = Sequential()
    model.add(Conv3D(4, kernel_size=(3, 3, 3), input_shape=(
        X.shape[1:]), padding='same'))
    print('x.dytapy',X.dtype)
    model.add(Activation('relu'))
    model.add(Conv3D(4, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.4))

    model.add(Conv3D(8, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(8, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25,.25]], gamma=2)],
                  optimizer=Adam(), metrics=['accuracy']
                  )
    model.summary()

    # #(model, show_shapes=True,
    #            to_file=os.path.join(args.output, 'model.png'))

    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.2, random_state=43,stratify=Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_new, test_size=0.2, random_state=43, stratify=Y_new)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch,
                        epochs=args.epoch, verbose=1, shuffle=True)
    print('histry',history.history)
    # model.evaluate(X_test, Y_test, verbose=0)
    y_pred=model.predict(X_test)
    # print('prediction',y_pred)
    # print(np.array(y_pred).shape)
    # y_test=np.argmax(np.array(y_pred), axis=1)
    # print(y_test)
    # print(Y_test)
    f1=mean_pred(Y_test,y_pred)
    print('f1shape',f1)
    model_json = model.to_json()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, 'ucf101_3dcnnmodel.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(args.output, 'ucf101_3dcnnmodel.hd5'))

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    plot_history(history, args.output)
    save_history(history, args.output)


if __name__ == '__main__':
    main()
