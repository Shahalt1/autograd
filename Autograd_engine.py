import math
from graphviz import Digraph
import random


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value (data={self.data})"

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):  # other / self
        return self * (other**-1)

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "Only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad = other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        self._backward = _backward
        return out

    def backward(self):
        stack = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                stack.append(node)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(stack):
            node._backward()


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

    def draw_dot(self, root, filename="graph", file_format="svg"):
        dot = Digraph(format=file_format, graph_attr={"rankdir": "LR"})

        nodes, edges = self.trace(root)
        for n in nodes:
            uid = str(id(n))  # data node id
            dot.node(
                name=uid,
                label="{%s | data %.2f | grad %.2f}" % (n.label, n.data, n.grad),
                shape="record",
            )
            if n._op:
                id1 = uid + n._op  # operator node id
                dot.node(name=id1, label=n._op)
                dot.edge(id1, uid)  # from operator node to data node

        for n1, n2 in edges:
            dot.edge(
                str(id(n1)), str(id(n2)) + n2._op
            )  # from data node to operator node

        dot.render(
            filename, format=file_format, cleanup=True
        )  # Render the graph to a file and cleanup temporary files
        return dot

    def trace(self, root):
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        build(root)
        return nodes, edges


class Neuron(Module):

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1), label=f"w{i}") for i in range(nin)]
        self.b = Value(random.uniform(-1, 1), label="b")

    def __call__(self, x):
        # w*x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        act.label = "wi*xi"
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [
            n(x) for n in self.neurons
        ]  # Neuron1([1,2]), Neuron2([1,2]) if self.neuron = [Neuron1, Neuron2]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # return [ p for neuron in self.neurons for p in neuron.parameters() ]
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params


# The __init__ function of the MLP class is only responsible for creating and initializing the layers of the network. The actual forward pass through the
# network, where n(x) is called for each neuron, happens in the __call__ function of the MLP class.


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [l for layer in self.layers for l in layer.parameters()]
