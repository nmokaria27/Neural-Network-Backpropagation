# Neural Network Backpropagation from Scratch

This project is a deep dive into the mechanics of neural networks, inspired by Andrej Karpathy's renowned "The spelled-out intro to neural networks and backpropagation: building micrograd" lecture. It involves building a small-scale automatic differentiation (autograd) engine from the ground up to gain a fundamental understanding of how backpropagation works.

## Core Concepts Implemented

The engine and notebooks demonstrate the core principles of training a neural network:

1.  **The `Value` Object**: The heart of the engine. It's a custom class that wraps a single scalar value and tracks its computational history. This allows it to store not just its data, but also its gradient (`grad`) and the operation (`_op`) that created it.

2.  **Computation Graph**: By linking `Value` objects together through mathematical operations (`+`, `*`, `relu`, etc.), we dynamically build a graph that represents our mathematical expression (i.e., the neural network).

3.  **Backpropagation**: The project implements a `backward()` method that traverses the computation graph from end to start, applying the **chain rule** at every step to automatically calculate the gradient of the final output with respect to every intermediate value and input.

![alt text](https://github.com/nmokaria27/Neural-Network-Backpropagation/blob/main/gd.svg)


4.  **Neural Network Components**: Built on top of the `Value` engine are simple, Pythonic classes for:
    *   `Neuron`: The basic computational unit.
    *   `Layer`: A collection of neurons.
    *   `MLP` (Multi-Layer Perceptron): The full neural network model, composed of multiple layers.

## Files in this Repository

*   `engine/`: This directory contains the core logic of the project.
    *   `engine.py`: Defines the `Value` object and its mathematical operations, including the `backward()` pass logic.
    *   `nn.py`: Defines the `Neuron`, `Layer`, and `MLP` classes.

*   `NeuralNetwork-Backpropagation.ipynb`: A Jupyter Notebook that provides a step-by-step, manual walkthrough of backpropagation on a simple expression. This is excellent for building initial intuition.

*   `MLP-Backprop.ipynb`: A Jupyter Notebook that uses the complete `engine` to build and train a Multi-Layer Perceptron on a non-linear "two moons" dataset. It demonstrates the full, automated training loop, including the forward pass, backward pass, and parameter updates (gradient descent).

## How to Use

To explore this project, it's recommended to follow the notebooks in order:

1.  Start with `NeuralNetwork-Backpropagation.ipynb` to understand the fundamentals of the chain rule and manual gradient calculation.
2.  Move on to `MLP-Backprop.ipynb` to see how the automated engine is used to train a complete neural network to solve a classification problem.

![alt text](https://github.com/nmokaria27/Neural-Network-Backpropagation/blob/main/two-moon.png)
