import numpy as np


def linear(z):
    return z


def linear_prime(a):
    return 1.


def relu(z):
    return np.maximum(z, 0)


def relu_prime(a):
    return (a > 0.).astype(float)


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def sigmoid_prime(a):
    return a * (1. - a)


def tanh(z):
    return np.tanh(z)


def tanh_prime(a):
    return 1. - a ** 2


def cost(a, y):
    return 0.5 * np.mean((a - y) ** 2)


def dcost(a, y):
    return (a - y)


def l2(layers):
    return sum(np.sum(l.w ** 2) for l in layers)


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


def backward(layers, da, cache, learning_rate):
    for layer in reversed(layers):
        da = layer.backward(da, *cache[-2:], learning_rate)
        cache.pop()


def fit(layers, x, y, epochs, batch_size, learning_rate):
    a = np.zeros_like(y)
    m = a.shape[1]

    for epoch in range(epochs):
        for j in range(0, m, batch_size):
            i = slice(j, j + batch_size)
            xi = x[:, i]
            yi = y[:, i]
            cache = [xi]
            ai = a[:, i] = forward(layers, xi, cache)
            backward(layers, dcost(ai, yi), cache, learning_rate)
        yield a

        c = cost(a, y)
        if c < 1e-4:
            break

    print(epoch, c)


class Dense:
    def __init__(self, in_features, out_features, activation='tanh'):
        self.in_features = in_features
        self.out_features = out_features

        self.g = globals()[activation]
        self.g_prime = globals()[activation+'_prime']

        self.reset()


    def reset(self):
        self.w = np.sqrt(1/self.in_features) * np.random.randn(self.out_features, self.in_features)
        self.b = np.sqrt(1/self.in_features) * np.random.randn(self.out_features, 1)
        self.m = 0
        self.v = 0


    def forward(self, a):
        z = self.w @ a + self.b
        a = self.g(z)
        return a


    def backward(self, da, a_prev, a, learning_rate):
        dz = da * self.g_prime(a)
        da = self.w.T @ dz

        dw = 1/dz.shape[1] * dz @ a_prev.T
        db = 1/dz.shape[1] * np.sum(dz, axis=1, keepdims=True)

        self.m = 0.9 * self.m + (1. - 0.9) * dw
        self.v = 0.999 * self.v + (1. - 0.999) * dw ** 2

        self.w -= learning_rate * self.m / (np.sqrt(self.v) + 1e-8)
        self.b -= learning_rate * db

        return da
