# Medium Problem: Concentric Circles Classification
Non-linear binary classification task on two concentric circles.

## Problem Description
The goal is to classify each point into one of two classes based on two input features:

- x
- y

The two output classes are:
- 0: outer circle
- 1: inner circle

## Dataset
The dataset is provided as `concentric_circles_train.csv` & `concentric_circles_test.csv`.

- Total samples: 1000
- Train/test split: 80% / 20%
- Noise: 0.1
- Inner circle factor: 0.5
- Random seed: 42

### Columns
- x
- y
- label

## Suggested Network Topology
- Input layer: 2 neurons
- Hidden layer 1: 8 neurons
- Hidden layer 2: 8 neurons
- Output layer: 1 neuron

Suggested topology: **2-8-8-1**

## Task Type
Binary classification

The task is used as the medium benchmark because the decision boundary has to separate an inner circle from an outer circle. This makes a purely linear solution insufficient and exposes differences between activation functions more clearly than the easy task.

## Suggested Activation Functions
Choose from the defined benchmark set:
- ReLU
- GeLU
- Sigmoid
- TanH
- Swish
- Identity

## Activation Function Placement
- Input layer: no activation function
- Hidden layers: selected activation function
- Output layer: Sigmoid

## Training
The number of epochs: 150
