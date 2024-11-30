import random
from data_structures import Value
from loss import calc_loss
from visaulization_model import draw_graph


class Module:
    pass

class Neuron(Module):
    def __init__(self, number_of_innodes: int) -> None:
        """
        creates neurons and their corresponding operations
        numbers_of_nodes: number of neurons to be created and initializes them
        """
        self.w = [Value(random.uniform(-1, 1)) for i in range(number_of_innodes)]
        self.b = Value(0.0001)

    def __call__(self, data: list):
        """
        Input data would be given to a neuron which it then returns the final calculation
        data can be list or scalar
        """
        #assert for type errors and change nested loops to lislt comprehension
        sum_ = 0
        for i in range(len(data)):
            sum_ += data[i] * self.w[i]
            #print(sum_)
        
        sum_ += self.b
        #print(f"result before activation: {sum_}")
        out = sum_.activation() #you can pass activation function name of your choice
        #print(f"result after activation: {out}")

        return out

    def __repr__(self):
        return f"{self.w}"

class Layer(Module):
    def __init__(self, n_neurons, ins) -> None:
        self.layer = [Neuron(ins) for i in range(n_neurons)]

    def __call__(self):
        pass

    def __repr__(self):
        return f"{[i for i in self.layer]}"

class MLP(Module):
    def __init__(self, d_input: int, n_neurons_in_layers: list):
        #if len(n_neurons_in_layers) != n_layers:
            #pass #assert error
        self.d_input = d_input
        self.n_neurons_in_layers = n_neurons_in_layers
        self.r = []
        p = [d_input] + n_neurons_in_layers + [1]
        for i in range(len(p)-1):
            t = Layer(p[i+1], p[i])
            self.r.append(t)
            

    def __call__(self, data):
        t = data
        #print(self.r)
        for i in range(len(self.r)):
            list_= []
            #print(f"len[0]={len(self.r[0].layer)}")
            for j in range(len(self.r[i].layer)):
                print(f"Input data: {t}")
                print(f"layer {i} neurons weights are: {self.r[i].layer}")
                list_.append(self.r[i].layer[j](t))
                #print(f"list_: {list_}")
            t = list_
        
        #return t[0].data
        return t[0] #we need them to be of type Value to be able to backpropagate through loss



'''
x = [4, 6]
mlp = MLP(2, [3,2])
y_pred = mlp(x)

print(y_pred)

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [1.0, 1.0, -1.0]
]

n = MLP(3, [4, 4, 1])
ys = [1.0, -1.0, 1.0]
ypred = [n(x) for x in xs]
print(ypred)

'''
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, 1.0]

network = MLP(3, [4, 4, 1])
ypred = [network(x) for x in xs]

print(f'prediction: {ypred}')
print(f'labels: {ys}')



loss = calc_loss(ys, ypred)
print(f"loss is: {loss}")
draw_graph(loss)

loss.grad = 1

loss.backward()

print(network.r[0].layer[1].w[1].grad)