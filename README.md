# Autograd NN
This neural network (NN) includes a `Neuron` class with weights, bias, and tanh activation. The `Layer` class groups neurons into a layer, and the `MLP` (Multi-Layer Perceptron) class stacks layers to form a fully connected feedforward network. It supports forward passes and visualizes the computational graph.

This neural network implementation is inspired by Andrej Karpathy's "micrograd" project. Karpathy's minimalist approach to building neural networks from scratch provided a deep understanding of the fundamental principles of neural computation and backpropagation. Extending his work, this project adds features such as visualizing the computational graph. I am deeply grateful for Karpathy's contributions, which have significantly advanced my learning and passion for neural networks.

# To use this neural network model, follow these steps:

1. **Initialize the Network**: Create an instance of the `MLP` class by specifying the number of input features and the architecture. For example, `mlp = MLP(2, [2, 1])` creates a network with 2 input features, one hidden layer with 2 neurons, and an output layer with 1 neuron.

2. **Forward Pass**: Pass input data through the network by calling the instance with the input, such as `output = mlp([Value(1.0, 'x1'), Value(2.0, 'x2')])`.

3. **Backward Pass**: Perform the backward pass to compute gradients by calling `output.backward()` on the output `Value` object.

4. **Zero Gradients**: Reset the gradients to zero before the next forward pass by calling `mlp.zero_grad()`.

5. **Visualize the Computational Graph**: Generate and save the computational graph to a file using `mlp.draw_dot(output, filename='mlp_graph', file_format='svg')`.

This will create an SVG file named `mlp_graph.svg` that visualizes the forward pass and gradients in the network.
