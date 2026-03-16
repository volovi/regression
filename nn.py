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


def reset(layers, optimizer):
    for layer in layers:
        layer.reset()

    optimizer.reset()


def parameters(layers):
    return [p for layer in layers for p in layer.parameters()]


def predict(layers, a):
    for layer in layers:
        a = layer.forward(a)
    return a


def forward(layers, a, cache):
    for layer in layers:
        a = layer.forward(a)
        cache.append(a)
    return a


def backward(layers, da, cache):
    for layer in reversed(layers):
        da = layer.backward(da, *cache[-2:])
        cache.pop()


def fit(layers, x, y, epochs, batch_size, optimizer):
    a = np.zeros_like(y)
    m = a.shape[1]

    for epoch in range(epochs):
        for j in range(0, m, batch_size):
            i = slice(j, j + batch_size)
            xi = x[:, i]
            yi = y[:, i]
            cache = [xi]
            ai = a[:, i] = forward(layers, xi, cache)
            backward(layers, loss_grad(ai, yi), cache)
            optimizer.step()
        yield a

        l = loss(a, y)
        if l < 1e-4:
            break

    print(epoch, l)


class Dense:
    def __init__(self, in_features, out_features, activation='tanh'):
        self.w, self.dw = self.p1 = np.zeros((2, out_features, in_features))
        self.b, self.db = self.p2 = np.zeros((2, out_features, 1))

        self.g = globals()[activation]
        self.g_grad = globals()[activation+'_grad']

        self.reset()


    def reset(self):
        for p in self.parameters():
            p[0] = np.random.standard_normal(p[0].shape) * np.sqrt(1./p[0].shape[1])
            p[1] = 0


    def forward(self, a):
        z = self.w @ a + self.b
        a = self.g(z)
        return a


    def backward(self, da, a_prev, a):
        dz = da * self.g_grad(a)
        da = self.w.T @ dz

        self.dw[:] = dz @ a_prev.T
        self.db[:] = np.sum(dz, axis=1, keepdims=True)
        return da


    def parameters(self):
        return [self.p1, self.p2]


class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.9, nesterov=False):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov

        self.reset()


    def step(self):
        for i, (p, g) in enumerate(self.parameters):
            v_new = self.momentum * self.v[i] + self.lr * g
            p -= -self.momentum * self.v[i] + (1 + self.momentum) * v_new if self.nesterov else v_new
            self.v[i] = v_new


    def reset(self):
        self.v = [np.zeros_like(p[0]) for p in self.parameters]
