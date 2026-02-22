import random
from MLP.engine import Value

class Module:
    """Base class for all neural network components."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    """A single neuron with optional ReLU activation."""

    def __init__(self, nin, nonlin=True):
        self.w = []
        for _ in range(nin):
            self.w.append(Value(random.uniform(-1, 1)))
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + (wi * xi)
        return act.relu() if self.nonlin else act

    def parameters(self):
        params = []
        for weight in self.w:
            params.append(weight)
        params.append(self.b)
        return params

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    """A layer made of multiple neurons."""

    def __init__(self, nin, nout, **kwargs):
        self.neurons = []
        for _ in range(nout):
            self.neurons.append(Neuron(nin, **kwargs))

    def __call__(self, x):
        out = []
        for neuron in self.neurons:
            out.append(neuron(x))
        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = []
        for neuron in self.neurons:
            neuron_params = neuron.parameters()
            for p in neuron_params:
                params.append(p)
        return params

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """A simple multi-layer perceptron."""

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            is_last_layer = i == len(nouts) - 1
            self.layers.append(
                Layer(sz[i], sz[i + 1], nonlin=not is_last_layer)
            )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            layer_params = layer.parameters()
            for p in layer_params:
                params.append(p)
        return params

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
