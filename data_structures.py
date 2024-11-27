import pandas as pd

class Value:
    def __init__(self, data: float, grad=0, _children=(), _op='') -> None:
        self.data = data
        self.grad = grad
        self._prev = set(_children)
        self._op = _op

    def __mul__(self, other):
        pass

    def __add__(self, other):
        pass

    def __pow__(self, other):
        pass

    def __repr__(self):
        print(f"Value({self.data})")
