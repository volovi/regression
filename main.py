import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nn

lr = 0.01
num = 4_000
epochs = 1_000
batch_size = 20
accu_steps = 2
momentum = 0.9
nesterov = True


def get_data():
    X = np.random.randn(2, num)
    return X, (X**2).sum(axis=0, keepdims=True)


def get_paraboloid_data(r):
    R, T = np.meshgrid(np.linspace(0, r), np.linspace(0, 2 * np.pi))
    X, Y = R * np.cos(T), R * np.sin(T)
    return X, Y, X**2 + Y**2


def inc(obj, stop, n=20):
    step = (stop - obj) / n

    for i in range(n):
        obj += step
        yield i


def frames():
    nn.reset(layers, opt)

    x, y = get_data()
    it = nn.fit(layers, opt, x, y, epochs, batch_size, accu_steps)

    yield from (Y for _ in zip(inc(X, x), inc(Y, next(it))))
    yield from it

    Y[:] = y


def func(a):
    surf.set_data_3d(*X, *a)
    scat.set_offsets(X.T)
    scat.set_array(*a)
    scat.set_sizes(np.exp2(*a))
  
    return surf, scat


def init_func():
    return surf, scat


fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection='3d', xlim=(-4, 4), ylim=(-4, 4))
ax2 = fig.add_subplot(122, xlim=(-4, 4), ylim=(-4, 4))
X, Y = get_data()
surf, = ax.plot(*X, *Y, '.')
scat = ax2.scatter(*X, s=np.exp2(*Y), c=Y[0], alpha=0.5, edgecolors='none')
ax.plot_wireframe(*get_paraboloid_data(r=4.5), linewidth=0.1, color='C7')
ax.view_init(elev=5, azim=-90, roll=0)
ax.set_aspect('equal')
ax.axis('off')
ax2.axis('off')
plt.tight_layout()

layers = [ nn.Dense(2, 64)
         , nn.Dense(64, 64)
         , nn.Dense(64, 1, activation='linear')
         ]

opt = nn.SGD(nn.parameters(layers), lr, momentum, nesterov)

ani = animation.FuncAnimation(fig, func, frames, init_func, cache_frame_data=False, interval=50, blit=True)
plt.show()
