import os
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

# カレントディレクトリのパス
CURRENT_DIR = os.getcwd()
# データを保存するディレクトリのパス
DATA_DIR = os.path.join(CURRENT_DIR, 'data/')
# ラベルの種類
NUM_LABEL = 10
# テストデータの割合
TEST_SIZE = 0.2
# ランダムシード値
RANDOM_STATE = 2
# シグモイド関数のパラメータ
ALPHA = 4.0
# 重みの初期値の大きさ
W_AMPLITUDE = 0.1
# 学習係数
LEARNING_RATE = 0.5
# バッチ数
BATCH_SIZE = 100
# エポック数
NUM_EPOCH = 20


class Sigmoid:
    """ シグモイド関数 """
    def __init__(self, alpha: float=ALPHA) -> None:
        self.alpha = alpha
        self.y = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ xとyは同形 """
        y = 1.0 / (1 + np.exp(-self.alpha * x))
        self.y = y
        return y

    def backward(self) -> np.ndarray:
        return self.y * (1.0 - self.y)


class ReLU:
    """ ReLU関数 """
    def __init__(self) -> None:
        self.x = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0, x)

    def backward(self) -> np.ndarray:
        return np.where(self.x > 0, 1, 0)


class Softmax:
    """ ソフトマックス関数 """
    def __init__(self) -> None:
        self.y = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ xとyは同形 """
        exp_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
        y = exp_x / np.sum(exp_x, axis=1).reshape(-1, 1)
        self.y = y
        return y


class Linear:
    """ 線形層の処理 """
    def __init__(self, in_dim: int, out_dim: int, activation) -> None:
        """
        @param:
            in_dim:  入力のユニット数
            out_dim: 出力のユニット数
            activation: 活性化関数
        """
        self.W = np.random.randn(in_dim, out_dim) / np.sqrt(in_dim)
        self.b = np.zeros(out_dim)
        self.activation = activation()
        self.delta = None
        self.x = None
        self.dW = None
        self.db = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # 順伝播計算
        self.x = x
        u = np.dot(x, self.W) + self.b
        self.z = self.activation(u)
        return self.z

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        @param:
            dout: 1つ上の層からの出力
        @return:
            dout: この層の出力
        """
        # 誤差計算
        self.delta = self.activation.backward() * dout
        dout = np.dot(self.delta, self.W.T)
        # 勾配計算
        self.dW = np.dot(self.x.T, self.delta)
        self.db = np.dot(np.ones(len(self.delta)), self.delta)
        return dout


class MLP:
    """ 多層パーセプトロン """
    def __init__(self, layers) -> None:
        self.layers = layers

    def train(self, x: np.ndarray, t: np.ndarray, lr: float=LEARNING_RATE) -> float:
        # 1. 順伝播
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)
        # 2. 損失関数の計算
        self.loss = np.sum(-t * np.log(self.y + 1e-7)) / len(x)
        # 3. 誤差逆伝播
        # 3.1. 最終層
        # 3.1.1. 最終層の誤差・勾配計算
        delta = (self.y - t) / len(self.layers[-1].x)
        self.layers[-1].delta = delta
        self.layers[-1].dW = np.dot(self.layers[-1].x.T, self.layers[-1].delta)
        self.layers[-1].db = np.dot(np.ones(len(self.layers[-1].x)), self.layers[-1].delta)
        dout = np.dot(self.layers[-1].delta, self.layers[-1].W.T)
        # 3.1.2. 最終層のパラメータ更新
        self.layers[-1].W -= lr * self.layers[-1].dW  # self.layers[-1].dW を用いて最終層の重みを更新しよう
        self.layers[-1].b -= lr * self.layers[-1].db  # self.layers[-1].db を用いて最終層のバイアスを更新しよう
        # 3.2. 中間層
        for layer in self.layers[-2::-1]:
            # 3.2.1. 中間層の誤差・勾配計算
            dout = layer.backward(dout)  # 逆伝播計算を順番に実行しよう
            # 3.2.2. パラメータの更新
            layer.W -= lr * layer.dW # 各層の重みを更新
            layer.b -= lr * layer.db  # 各層のバイアスを更新
        return self.loss

    def test(self, x: np.ndarray, t: np.ndarray) -> float:
        # 性能をテストデータで調べるために用いる
        # よって、誤差逆伝播は不要
        # 順伝播 (train関数と同様)
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)
        self.loss = np.sum(-t*np.log(self.y + 1e-7)) / len(x)
        return self.loss


def main() -> None:
    """ main関数 """
    X, y = load_data()
    X_train, X_test, Y_train, Y_test = reshape_data(X, y)
    model = MLP([Linear(X.shape[1], 1000, Sigmoid),
                 Linear(1000, 1000, Sigmoid),
                 Linear(1000, NUM_LABEL, Softmax)
                ])
    for epoch in range(NUM_EPOCH):
        index = np.random.permutation(len(X_train))
        model, loss_train, accuracy_train = train_test_model(model, X_train, Y_train, index, mode='train')
        model, loss_test, accuracy_test = train_test_model(model, X_test, Y_test, index, mode='test')
        print('{} / {} epoch | TRAIN loss: {:.4f}, accuracy: {:.2%} | TEST loss: {:.4f}, accuracy: {:.2%}'.format(epoch+1, NUM_EPOCH, loss_train, accuracy_train, loss_test, accuracy_test))


def load_data(data_dir: str=DATA_DIR) -> (np.ndarray, np.ndarray):
    """
    MNISTデータのロード
    @param:
        data_dir: ロードしたデータを保存するディレクトリパス (= DATA_DIR)
    @return:
        X: 説明変数 (7000, 784)
        y: 目的変数 (7000,)
    """
    mnist = fetch_mldata('MNIST original', data_home=data_dir)
    X, y = mnist.data, mnist.target
    X = X / 255.  # 説明変数を正規化
    y = y.astype('int')
    return X, y


def reshape_data(X: np.ndarray, y: np.ndarray, test_size: float=TEST_SIZE, random_state: int=RANDOM_STATE,
                 num_label: int=NUM_LABEL) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    データを整形
    @param:
        X: 説明変数 (7000, 784)
        y: 目的変数 (7000,)
        test_size: テストデータの割合  (= TEST_SIZE)
        random_state: ランダムシード値 (= RANDOM_STATE)
        num_label: ラベルの種類        (= NUM_LABEL)
    @return:
        X_train: 訓練データ説明変数  (7000*(1-test_size), 784)
        X_test: テストデータ説明変数 (7000*test_size, 784)
        Y_train: 訓練データ目的変数  (7000*(1-test_size), 10)
        Y_test: テストデータ目的変数 (7000*test_size, 10)
    """
    # データを訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    Y_train = np.eye(NUM_LABEL)[y_train].astype(np.int32)
    Y_test = np.eye(NUM_LABEL)[y_test].astype(np.int32)
    return X_train, X_test, Y_train, Y_test


def train_test_model(model, X: np.ndarray, Y: np.ndarray, index: np.ndarray,
                     batch_size: int=BATCH_SIZE, mode: str='train') -> (MLP, float, float):
    sum_loss = 0
    y_pred = []
    for i in range(0, len(X), batch_size):
        if mode == 'train':
            X_ = X[index[i:i+batch_size]]
            T_ = Y[index[i:i+batch_size]]
            sum_loss += model.train(X_, T_) * len(X_)
        else:
            X_ = X[i:i+batch_size]
            T_ = Y[i:i+batch_size]
            sum_loss += model.test(X_, T_) * len(X_)
        y_pred.extend(np.argmax(model.y, axis=1))
    loss = sum_loss / len(X)
    if mode == 'train':
        accuracy = np.sum(np.eye(NUM_LABEL)[y_pred] * Y[index]) / len(X)
    else:
        accuracy = np.sum(np.eye(NUM_LABEL)[y_pred] * Y) / len(X)
    return model, loss, accuracy


if __name__ == '__main__':
    main()
