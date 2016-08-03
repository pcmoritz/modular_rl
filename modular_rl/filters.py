from .running_stat import RunningStat
import numpy as np

class Composition(object):
    def __init__(self, fs):
        self.fs = fs
    def __call__(self, x, update=True):
        for f in self.fs:
            x = f(x)
        return x
    def output_shape(self, input_space):
        out = input_space.shape
        for f in self.fs:
            out = f.output_shape(out)
        return out

class ZFilter(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std+1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    def output_shape(self, input_space):
        return input_space.shape

    @staticmethod
    def deserialize(data):
        demean, destd, clip = data["demean"], data["destd"], data["clip"]
        n, M, S = data["n"], data["M"], data["S"]
        filt = ZFilter(M.shape, demean, destd, clip)
        filt.rs = RunningStat(M.shape)
        filt.rs._n = n
        filt.rs._M = M
        filt.rs._S = S
        return filt

    def serialize(self):
        return {"demean": self.demean, "destd": self.destd, "clip": self.clip,
                "n": self.rs._n, "M": self.rs._M, "S": self.rs._S}

class Flatten(object):
    def __call__(self, x, update=True):
        return x.ravel()
    def output_shape(self, input_space):
        return (int(np.prod(input_space.shape)),)

class Ind2OneHot(object):
    def __init__(self, n):
        self.n = n
    def __call__(self, x, update=True):
        out = np.zeros(self.n)
        out[x] = 1
        return out
    def output_shape(self, input_space):
        return (input_space.n,)
