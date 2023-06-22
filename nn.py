import random
from engine import Value

class Neuron:
    def __init__(self,nin): 
        '''
        nin : no. of inputs
        '''
        self.w = [Value(random.uniform(-1,1)) for i in range(nin)]
        self.b = Value(random.uniform(-1,1))
        
    def __call__(self,x):
        #  w*x +b
        z = (sum([wi*xi for wi,xi in zip(self.w,x)],self.b))
        op = z.tanh()
        return op
    
    def parameters(self):
        return self.w + [self.b]
    
    
class Layer:
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for i in range(nout)]
        
    def __call__(self,x):
        z = [n(x) for n in self.neurons]
        return z
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        # params = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters
        #     params.extend(ps)
            
        # return params
        
        
class MLP:
    def __init__(self,nin,nouts): 
        """
        nin is the no of the inputs 
        and nouts is the list of layers with thier respective no. of neurons in them
        """
        self.sz = [nin] + nouts
        self.layers = [Layer(self.sz[i] , self.sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]