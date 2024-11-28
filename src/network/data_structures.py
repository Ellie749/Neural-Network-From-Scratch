import pandas as pd

class Value:
    def __init__(self, data: float, grad=0, _children=(), _op='') -> None:
        self.data = data
        self.grad = grad
        self._prev = set(_children)
        self._op = _op

    def __mul__(self, other):
        
        def backward(self, other):
            self.grad += other.data
            other.grad += self.data

        return Value(self.data * other.data, _children=(self, other), _op='*')

    def __add__(self, other):

        def backward(self, other):
            self.grad += self.grad
            other.grad += other.grad
        return Value(self.data + other.data, _children=(self, other), _op='+')

    def __pow__(self, other):
        pass

    def __repr__(self):
        print(f"Value({self.data})")
