import numpy as np


def linear(z):
    return z


def linear_grad(a):
    return 1.


def relu(z):
    return np.maximum(z, 0)


def relu_grad(a):
    return (a > 0.).astype(float)


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def sigmoid_grad(a):
    return a * (1. - a)


def tanh(z):
    return np.tanh(z)


def tanh_grad(a):
    return 1. - a ** 2


def loss(a, y):
    return 0.5 * np.mean((a - y) ** 2)


def loss_grad(a, y):
    return (a - y) / a.shape[1]


def reset(layers):
    for layer in layers:
        layer.reset()


def predict(layers, a):
    for layer in layers:
        a = layer.forward(a)
    return a


def forward(layers, a, cache):
    for layer in layers:
        a = layer.forward(a)
        cache.append(a)
    return a


def backward(layers, da, cache, lr):
    for layer in reversed(layers):
        da = layer.backward(da, *cache[-2:], lr)
        cache.pop()


def fit(layers, x, y, epochs, batch_size, lr):
    a = np.zeros_like(y)
    m = a.shape[1]

    for epoch in range(epochs):
        for j in range(0, m, batch_size):
            i = slice(j, j + batch_size)
            xi = x[:, i]
            yi = y[:, i]
            cache = [xi]
            ai = a[:, i] = forward(layers, xi, cache)
            backward(layers, loss_grad(ai, yi), cache, lr)
        yield a

        l = loss(a, y)
        if l < 1e-4:
            break

    print(epoch, l)


class Dense:
    def __init__(self, din, dout, activation='tanh'):
        self.din = din
        self.dout = dout

        self.g = globals()[activation]
        self.g_grad = globals()[activation+'_grad']

        self.reset()


    def reset(self):
        self.w = np.sqrt(1/self.din) * np.random.randn(self.dout, self.din)
        self.b = np.sqrt(1/self.din) * np.random.randn(self.dout, 1)
        self.m = 0
        self.v = 0


    def forward(self, a):
        z = self.w @ a + self.b
        a = self.g(z)
        return a


    def backward(self, da, a_prev, a, lr):
        dz = da * self.g_grad(a)
        da = self.w.T @ dz

        dw = dz @ a_prev.T
        db = np.sum(dz, axis=1, keepdims=True)

        self.m = 0.9 * self.m + (1. - 0.9) * dw
        self.v = 0.999 * self.v + (1. - 0.999) * dw ** 2

        self.w -= lr * self.m / (np.sqrt(self.v) + 1e-8)
        self.b -= lr * db

        return da
