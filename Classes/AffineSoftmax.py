from Classes.Library import np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self. y = None
        self.t = None

    def softmax(self, x):
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x

        return y

    def cross_entropy_error(self, y, t):
        delta = 1e-7

        return -np.sum(t*np.log(y+delta))

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size

        return dx