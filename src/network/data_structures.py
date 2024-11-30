import math
import pandas as pd
from activation import Activation
from visaulization_model import draw_graph

class Value:
    def __init__(self, data: float, grad=0, _children=(), _op='') -> None:
        """
        Initialize object values

        data: holds value of the object which mainly is int or float
        grad: holds the gradients which are calculated during backpropagation
        _prev: in case of arithmatic calculations the source nodes are added to the children to keep track of connections
        _op: arithmetic operation in nodes - for leaf nodes it is None
        _backward: keeps track of the relevant derivative of the node relevant to the arithmetic operation
        """
        self.data = data
        self.grad = grad
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None #function to be called

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward 
        #backward is a function (iterator) which should be called during backpropagation and not in here
        #it does not assign the gradient of the node itself, but the nodes that resulted in this node
        return out
    
    def __rmul__(self, other): 
        return self * other # you still have to define the operation

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out


    def backward(self):
        visited = set()
        nodes = []

        def build_back(node):
            """
            build backward graph
            """
            if node not in visited:
                nodes.append(node)
                visited.add(node)
                for child in node._prev:
                    build_back(child)
            return nodes
        all_nodes = build_back(self)

        self.grad =1
        for node in all_nodes:
            node._backward()

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return self - other
        
    def __neg__(self): #because of sub
        return self * -1

    """
    def __pow__(self, other):
        '''
        out = Value(0)
        for i in range(other):
            out += i*i
        return out
        '''
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        print(f"pow out: {out}")
        return out
    """

    def activation(self, name='tanh'):
        '''
        act = Activation(name)
        out = act.apply(self.data)
        return Value(out)
        '''
        out = Value(0 if self.data < 0 else self.data, (self,), _op='ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other): # Value(3) + 2 == 2 + Value(3)
        return self + other
    
    def __pow__(self, other):
        pass

    def __repr__(self):
        return f"Value({self.data})" # should return string and not print(string)


"""
#TEST CASE

def main():
    a = Value(3)
    b = Value(4)
    c = Value(1)
    print(b)
    
    d = a * b
    f = d + c
    print(d)

    f.grad = 1
    f.backward()

    print(b.grad)

if __name__ == "main":
    main()
"""
