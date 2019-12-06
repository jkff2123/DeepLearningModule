from Classes.Library import np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads, keys):
        for key in keys:
            for cnt in range(len(params[key])):
                params[key][cnt] -= self.lr * grads[key][cnt]

class adagrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads, keys):
        if self.h is None:
            self.h = {}
            for key in keys:
                self.h[key] = {}
                for cnt in range(len(params[key])):
                    self.h[key][cnt] = np.zeros_like(params[key][cnt])

        for key in keys:
            for cnt in range(len(params[key])):
                self.h[key][cnt] += grads[key][cnt] * grads[key][cnt]
                params[key][cnt] -= self.lr * grads[key][cnt] / (np.sqrt(self.h[key][cnt]) + 1e-7)
        
class adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
    
    def update(self, params, grads, keys):
        if self.m is None:
            self.m, self.v = {}, {}
            for key in keys:
                self.m[key] = {}
                self.v[key] = {}
                for cnt in range(len(params[key])):
                    self.m[key][cnt] = np.zeros_like(params[key][cnt])
                    self.v[key][cnt] = np.zeros_like(params[key][cnt])
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in keys:
            for cnt in range(len(params[key])):
                self.m[key][cnt] += (1 - self.beta1) * (grads[key][cnt] - self.m[key][cnt])
                self.v[key][cnt] += (1 - self.beta2) * (grads[key][cnt]**2 - self.v[key][cnt])

                params[key][cnt] -= lr_t * self.m[key][cnt] / (np.sqrt(self.v[key][cnt]) + 1e-7)