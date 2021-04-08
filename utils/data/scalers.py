from abc import ABC, abstractmethod


class Scaler(ABC):
    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def fit_and_transform(self, x):
        pass

    @abstractmethod
    def inverse_transform(self, x):
        pass


class StandardScaler(Scaler):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
    
    def transform(self, x):
        x -= self.mean
        x /= self.std
        return x

    def fit_and_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x *= self.std
        x += self.mean
        return x
