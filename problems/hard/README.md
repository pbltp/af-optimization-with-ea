# Hard Problem - Crossing Spirals Classification
Non-linear binary classification task on complex geometric patterns.

## Problem Description
The goal is to distinguish between two interlocking spirals where one spiral follows a standard path and the second spiral follows a modulated, "wobbling" path that creates multiple intersection nodes.

| No-noise  | With-noise |
| ------------- | ------------- |
| ![Crossing Spirals no noise](crossing_spirals_no_noise.png) | ![Crossing Spirals with noise](crossing_spirals_noise.png) |

The six input features are:
- x
- y
- x_sin: $\sin(x)$
- y_sin: $\sin(y)$
- x_cos: $\cos(x)$
- y_cos: $\cos(y)$

The two output classes are:
- 0 = Red Spiral
- 1 = Blue Spiral

## Dataset
The dataset is provided as `crossing_spirals_train.csv` & `crossing_spirals_test.csv`.

### Columns
- x
- y
- x_sin
- y_sin
- x_cos
- y_cos
- label

## Suggested Network Topology
- Input layer: 6 neurons
- Hidden layer 1: 16 neurons
- Hidden layer 2: 16 neurons
- Output layer: 1 neuron 

Suggested topology: **6-16-16-1**

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

## Training
The number of epochs: 250
