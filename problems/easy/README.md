# Easy Problem: Two Moons Classification
Non-linear binary classification task on two interleaving moon-shaped classes.

## Problem Description
The goal is to classify each point into one of two classes based on two input features:

- x
- y

The two output classes are:
- 0: first moon
- 1: second moon

## Dataset
The dataset is provided as `two_moons_train.csv` & `two_moons_test.csv`.

- Total samples: 1000
- Train/test split: 80% / 20%
- Noise: 0.2
- Random seed: 42

### Columns
- x
- y
- label

## Suggested Network Topology
- Input layer: 2 neurons
- Hidden layer: 8 neurons
- Output layer: 1 neuron

Suggested topology: **2-8-1**

## Task Type
Binary classification

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
- Hidden layer: selected activation function
- Output layer: Sigmoid

## Training
The number of epochs: 100
