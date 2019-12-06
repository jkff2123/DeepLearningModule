from collections import OrderedDict
from Classes.Library import np
from Classes.ActivatorLayers import Sigmoid, Relu
from Classes.AffineSoftmax import Affine, SoftmaxWithLoss
from Classes.Optimizer import adam

class NeuralNet:
    def __init__(self, dim, weight = 1.0, load = False):
        if type(dim) is not list:
            return

        self.dim = dim
        self.param = {}
        self.param["w"] = {}
        self.param["b"] = {}

        if load:
            return
        else:
            for cnt in range(len(dim) - 1):
                self.param["w"][cnt] = weight * np.random.randn(dim[cnt], dim[cnt+1])
                self.param["b"][cnt] = np.zeros(dim[cnt+1])

        self.layers = OrderedDict()
        for cnt in range(len(dim) - 1):
            idx = "af" + str(cnt)
            self.layers[idx] = Affine(self.param["w"][cnt], self.param["b"][cnt])
            if cnt < len(dim) - 2:
                idx = "relu" + str(cnt)
                self.layers[idx] = Relu()

        self.last = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t, one_hot_encoding):
        y = self.predict(x)

        return self.last.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accu = np.sum(y==t)/float(x.shape[0])

        return accu

    def gradient(self, x, t):
        self.loss(x, t, True)

        dout = 1
        dout = self.last.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["w"] = {}
        grads["b"] = {}
        for cnt in range(len(self.dim) - 1):
            idx = "af" + str(cnt)
            grads["w"][cnt] = self.layers[idx].dW
            grads["b"][cnt] = self.layers[idx].db

        return grads

    def neuralstudy(self, x, t, batch_size, iter_cnt, ln_rate=0.1):
        for i in range(iter_cnt):
            batch_mask = np.random.choice(x.shape[0], batch_size)
            x_batch = x[batch_mask]
            t_batch = t[batch_mask]

            grads = self.gradient(x_batch, t_batch)

            keys = ["w","b"]
            optimizer = adam(lr = ln_rate)
            optimizer.update(self.param, grads, keys)

            # Study check
            if ((i / iter_cnt) * 100) % 1 == 0:
                print(str((i / iter_cnt) * 100) + "% Accuracy: " + str(self.accuracy(x, t)))

        print("Train accuracy: " + str(self.accuracy(x, t)) + "\nStudy complete.")

    def neuraloutput(self, x):
        return self.predict(x)
