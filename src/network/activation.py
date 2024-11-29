import math

class Activation:
    def __init__(self, name: str):
        self.name = name

    def apply(self, data:float):
        
        #match self.name:   #requires Python > 3.10
            #case 'ReLU':
                #pass
        if self.name == 'ReLU':
            pass
        elif self.name == 'tanh':
            return (math.exp(data) - math.exp(-data)) / (math.exp(data) + math.exp(-data))
        elif self.name == 'sigmoid':
            pass
