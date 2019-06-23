import numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# 検証用データ割合
VALID_SIZE = 0.2
# ランダムシード値
RANDOM_STATE = 0
# 正規化定数
NORM_CONST = 255.0
# 学習係数
LEARNING_RATE = 1e-4
# エポック数
NUM_EPOCHS = 10


def main():
    """ main """
    train_images, test_images, train_labels, test_labels = load_dataset()
    model = make_cnn()
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_images, train_labels, epochs=NUM_EPOCHS, verbose=1, validation_split=VALID_SIZE)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test Accuracy = {:.2%}".format(test_acc))
    return


def make_cnn() -> models.Sequential:
    """
    CNNアーキテクチャを構築
    @return: CNN
    """
    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    print(model.summary())
    return model


def load_dataset() -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    CIFAR-10データセットの読み込み
    @return:
        train_images: 訓練データ画像
        test_images: テストデータ画像
        train_labels: 訓練データラベル
        test_labels: テストデータラベル
    """
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.astype("float32") / NORM_CONST
    test_images = test_images.astype("float32") / NORM_CONST
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    print("訓練データ   | 画像: {}, ラベル: {}".format(train_images.shape, train_labels.shape))
    print("テストデータ | 画像: {}, ラベル: {}".format(test_images.shape, test_labels.shape))
    return train_images, test_images, train_labels, test_labels


if __name__ == "__main__":
    main()
