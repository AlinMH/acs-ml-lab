import numpy as np
import mnist
from typing import List
from im2col import im2col_indices, col2im_indices


class Layer:

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, *args, **kwargs):
        pass  # If a layer has no parameters, then this function does nothing


class FeedForwardNetwork:

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self._inputs = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self._inputs = []
        for layer in self.layers:
            if train:
                self._inputs.append(x)
            x = layer.forward(x)
        return x

    def backward(self, dy: np.ndarray) -> np.ndarray:
        # TODO <0> : Compute the backward phase
        layer_count = len(self.layers)

        for i in range(layer_count - 1, 0, -1):
            dy = self.layers[i].backward(self._inputs[i], dy)

        del self._inputs

    def update(self, *args, **kwargs):
        for layer in self.layers:
            layer.update(*args, **kwargs)


class Linear(Layer):

    def __init__(self, insize: int, outsize: int) -> None:
        bound = np.sqrt(6. / insize)
        self.weight = np.random.uniform(-bound, bound, (insize, outsize))
        self.bias = np.zeros((outsize,))

        self.dweight = np.zeros_like(self.weight)
        self.dbias = np.zeros_like(self.bias)
        self.dw = np.zeros_like(self.weight)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO <1> : compute the output of a linear layer
        return (x @ self.weight) + self.bias

    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        # TODO <2> : compute dweight, dbias and  return dx
        self.dweight = x.T @ dy
        self.dbias = dy.sum(axis=0)

        return (dy @ self.weight.T)

    def update(self, mode='SGD', lr=0.001, mu=0.9):

        if mode == 'SGD':
            self.weight -= lr * self.dweight
            self.bias -= lr * self.dbias
        elif mode == 'momentum':
            self.dw = -lr * self.dweight + mu * self.dw
            self.weight += self.dw
        else:
            raise ValueError('mode should be SGD or momentum, not ' + str(mode))


class ReLU(Layer):
    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        return dy * (x > 0)


class MaxPool2D(Layer):
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def forward(self, x: np.ndarray) -> np.ndarray:
        def maxpool(X_col):
            max_idx = np.argmax(X_col, axis=0)
            out = X_col[max_idx, range(max_idx.size)]
            return out, max_idx

        x = x.reshape((x.shape[0], 1, 2 * x.shape[2], -1))
        n, d, h, w = x.shape
        h_out = (h - self.size) / self.stride + 1
        w_out = (w - self.size) / self.stride + 1

        if not w_out.is_integer() or not h_out.is_integer():
            raise Exception('Invalid output dimension!')

        h_out, w_out = int(h_out), int(w_out)
        X_reshaped = x.reshape(n * d, 1, h, w)
        X_col = im2col_indices(X_reshaped, self.size, self.size, padding=0, stride=self.stride)
        out, pool_cache = maxpool(X_col)
        out = out.reshape(h_out, w_out, n, d)
        out = out.transpose(2, 3, 0, 1)
        return out

    # needs rework
    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        def dmaxpool(dX_col, dout_col, pool_cache):
            dX_col[pool_cache, range(dout_col.size)] = dout_col
            return dX_col

        x = x.reshape((x.shape[0], 1, 2 * x.shape[2], -1))
        X_col = im2col_indices(x, self.size, self.size, padding=0, stride=self.stride)

        n, d, w, h = x.shape
        dX_col = np.zeros_like(X_col)
        dout_col = dy.transpose(2, 3, 0, 1).ravel()

        # dX = dmaxpool(dX_col, dout_col, pool_cache)
        dX = col2im_indices(dX_col, (n * d, 1, h, w), self.size, self.size, padding=0, stride=self.stride)
        dX = dX.reshape(x.shape)
        return dX

    def update(self, *args, **kwargs):
        pass


class Flatten(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], -1)

    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        return dy

    def update(self, *args, **kwargs):
        pass


class Conv2D(Layer):
    def __init__(self, filters, D, H, W, stride, padding):
        bound = np.sqrt(6. / filters)

        self.weight = np.random.uniform(-bound, bound, (filters, D, H, W))
        self.bias = np.zeros((filters,))

        self.stride = stride
        self.padding = padding

        self.dweight = np.zeros_like(self.weight)
        self.dbias = np.zeros_like(self.bias)
        self.dw = np.zeros_like(self.weight)

    def forward(self, x: np.ndarray) -> np.ndarray:
        n_filters, d_filter, h_filter, w_filter = self.weight.shape
        x = x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
        n_x, d_x, h_x, w_x = x.shape

        h_out = (h_x - h_filter + 2 * self.padding) / self.stride + 1
        w_out = (w_x - w_filter + 2 * self.padding) / self.stride + 1

        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception('Invalid output dimension!')

        h_out, w_out = int(h_out), int(w_out)
        X_col = im2col_indices(x, h_filter, w_filter, padding=self.padding, stride=self.stride)
        W_col = self.weight.reshape(n_filters, -1)

        out = W_col @ X_col
        out += np.array(out.shape[1] * [self.bias]).T
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        n_filter, d_filter, h_filter, w_filter = self.weight.shape
        db = np.sum(dy, axis=(0, 2, 3))
        self.dbias = db.reshape(n_filter, -1)

        X_col = im2col_indices(x, h_filter, w_filter, padding=self.padding, stride=self.stride)
        dout_reshaped = dy.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = dout_reshaped @ X_col.T
        self.dweight = dW.reshape(self.weight.shape)

        W_reshape = self.weight.reshape(n_filter, -1)
        dX_col = W_reshape.T @ dout_reshaped
        dX = col2im_indices(dX_col, x.shape, h_filter, w_filter, padding=self.padding, stride=self.stride)

        return dX

    def update(self, mode='SGD', lr=0.001, mu=0.9):
        self.weight -= lr * self.dweight
        self.bias -= lr * self.dbias


class CrossEntropy:

    def __init__(self):
        pass

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        # TODO <5> : Compute the negative log likelihood
        return (np.log(np.exp(y).sum(axis=1)) - y[np.arange(len(t)), t]).mean()

    def backward(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        # TODO <6> : Compute dl/dy
        t_size = len(t)

        z = np.zeros_like(y)
        z[np.arange(t_size), t] = 1

        P = np.exp(y) / np.exp(y).sum(axis=1, keepdims=True)

        return np.subtract(P, z) / t_size


def accuracy(y: np.ndarray, t: np.ndarray) -> float:
    # TODO <7> : Compute accuracy
    return np.mean([1 if np.argmax(y[i]) == t[i] else 0 for i in range(len(y))])


if __name__ == '__main__':
    train_imgs = mnist.train_images()
    train_labels = mnist.train_labels()
    test_imgs = mnist.test_images()
    test_labels = mnist.test_labels()

    WIDTH = 28
    HEIGHT = 28

    mean, std = train_imgs.mean(), train_imgs.std()
    train_imgs = (train_imgs - mean) / std
    test_imgs = (test_imgs - mean) / std

    BATCH_SIZE = 128
    HIDDEN_UNITS = 1682
    EPOCHS_NO = 20

    optimize_args = {'mode': 'momentum', 'lr': .005}

    net = FeedForwardNetwork([Conv2D(filters=2, D=1, H=2, W=2, stride=1, padding=1),
                              ReLU(),
                              Flatten(),
                              Linear(HIDDEN_UNITS, 10)])
    cost_function = CrossEntropy()

    for epoch in range(EPOCHS_NO):
        for b_no, idx in enumerate(range(0, len(train_imgs), BATCH_SIZE)):
            # 1. Prepare next batch
            x = train_imgs[idx:idx + BATCH_SIZE, :, :]
            t = train_labels[idx:idx + BATCH_SIZE]

            # 2. Compute gradient
            # TODO <8> : Compute gradient
            y = net.forward(x)
            loss = cost_function.forward(y, t)
            dl_dy = cost_function.backward(y, t)
            net.backward(dl_dy)

            # 3. Update network parameters
            net.update(**optimize_args)

            print(f'\rEpoch {epoch + 1:02d} '
                  f'| Batch {b_no:03d} '
                  f'| Train NLL: {loss:6.3f} '
                  f'| Train Acc: {accuracy(y, t) * 100:6.2f}% ', end='')

        y = net.forward(test_imgs, train=False)
        test_nll = cost_function.forward(y, test_labels)
        print(f'| Test NLL: {test_nll:6.3f} '
              f'| Test Acc: {accuracy(y, test_labels) * 100:3.2f}%')
