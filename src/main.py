#!/usr/bin/env python3
# Based on Basic Group's work

import csv
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RANDOM_SEED = 42
LEARNING_RATE = 0.01
BATCH_SIZE = 32

POPULATION_SIZE = 8
TOURNAMENT_SIZE = 2
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.01
MUTATION_SELECTION = 0.2

ACTIVATIONS = ["ReLU", "GeLU", "Sigmoid", "TanH", "Swish", "Identity"]

PROBLEMS = {
    "Easy": {
        "train": ROOT / "problems/easy/two_moons_train.csv",
        "test": ROOT / "problems/easy/two_moons_test.csv",
        "hidden_layers": [8],
        "epochs": 100,
    },
    "Medium": {
        "train": ROOT / "problems/medium/concentric_circles_train.csv",
        "test": ROOT / "problems/medium/concentric_circles_test.csv",
        "hidden_layers": [8, 8],
        "epochs": 150,
    },
    "Hard": {
        "train": ROOT / "problems/hard/crossing_spirals_train.csv",
        "test": ROOT / "problems/hard/crossing_spirals_test.csv",
        "hidden_layers": [16, 16],
        "epochs": 250,
    },
}


def load_dataset(path):
    with path.open() as csv_file:
        reader = csv.DictReader(csv_file)
        feature_names = [name for name in reader.fieldnames if name != "label"]
        features = []
        labels = []

        for row in reader:
            features.append([float(row[name]) for name in feature_names])
            labels.append(float(row["label"]))

    return feature_names, features, labels


def standardize(train_features, test_features):
    feature_count = len(train_features[0])
    means = []
    stds = []

    for col in range(feature_count):
        values = [row[col] for row in train_features]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        means.append(mean)
        stds.append(std if std > 0 else 1.0)

    def transform(features):
        return [
            [(row[col] - means[col]) / stds[col] for col in range(feature_count)]
            for row in features
        ]

    return transform(train_features), transform(test_features)


def relu(x):
    return x if x > 0 else 0.0

def gelu(x):
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def swish(x):
    return x * sigmoid(x)

def identity(x):
    return x

ActivationFunMap = {
    "ReLU": relu,
    "GeLU": gelu,
    "Sigmoid": sigmoid,
    "TanH": math.tanh,
    "Swish": swish,
    "Identity": identity
}


def relu_derived(x):
    return 1.0 if x > 0 else 0.0

def gelu_derived(x):
    normal_cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    normal_pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return normal_cdf + x * normal_pdf

def sigmoid_derived(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def tanh_derived(x):
    t = math.tanh(x)
    return 1.0 - t * t

def swish_derived(x):
    s = sigmoid(x)
    return s + x * s * (1.0 - s)

def identity_derived(x):
    return 1.0

ActivationDerivativeMap = {
    "ReLU": relu_derived,
    "GeLU": gelu_derived,
    "Sigmoid": sigmoid_derived,
    "TanH": tanh_derived,
    "Swish": swish_derived,
    "Identity": identity_derived
}

ActivationList = [
    "ReLU", "GeLU", "Sigmoid", "TanH", "Swish", "Identity"
]

def matmul_add_bias(matrix, weights, bias):
    output = []

    for row in matrix:
        out_row = []
        for out_col in range(len(bias)):
            value = bias[out_col]
            for in_col, row_value in enumerate(row):
                value += row_value * weights[in_col][out_col]
            out_row.append(value)
        output.append(out_row)

    return output

def apply_sigmoid(matrix):
    return [[sigmoid(value) for value in row] for row in matrix]

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def matmul(left, right):
    rows = len(left)
    inner = len(right)
    cols = len(right[0])
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]

    for row in range(rows):
        for mid in range(inner):
            left_value = left[row][mid]
            for col in range(cols):
                result[row][col] += left_value * right[mid][col]

    return result


def zeros_like(matrix):
    return [[0.0 for _ in row] for row in matrix]


def get_activation_funs(x):
    return [ActivationFunMap[x], ActivationDerivativeMap[x]]

class NeuralNetwork:
    def __init__(self, layer_sizes, rng, noinit = False):
        self.weights = []
        self.biases = []
        self.activation_funs = []
        if noinit:
            return

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            layer_weights = [
                [rng.uniform(-limit, limit) for _ in range(fan_out)]
                for _ in range(fan_in)
            ]
            layer_biases = [0.0 for _ in range(fan_out)]
            layer_activation = [
                get_activation_funs(rng.choice(ActivationList)) for _ in range(fan_out)
            ]
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)
            self.activation_funs.append(layer_activation)

    def crossover(self, fa, fb, rng):
        self.weights = fa.weights.copy()
        self.biases = fa.biases.copy()
        for layer in range(len(fa.activation_funs)):
            layer_activation = [
                a if rng.random() <= .5 else b for a, b in zip(
                    fa.activation_funs[layer], fb.activation_funs[layer]
                )
            ]
            self.activation_funs.append(layer_activation)

    def copy(self):
        res = NeuralNetwork(None, None, noinit = True)
        res.weights = self.weights.copy()
        res.biases = self.biases.copy()
        res.activation_funs = self.activation_funs.copy()
        return res

    def apply_activation_fun(self, x, i, col):
        fun = self.activation_funs[i][col][0]
        return fun(x)

    def apply_activation_derivative(self, x, i, col):
        fun = self.activation_funs[i][col][1]
        return fun(x)

    def apply_activations(self, matrix, i):
        return [
            [self.apply_activation_fun(value, i, ci) for ci, value in enumerate(row)]
            for row in matrix
        ]

    def forward(self, features):
        activations = [features]
        pre_activations = []
        current = features

        # XXX torchify: https://stackoverflow.com/a/78984724
        # I haven't gotten it to work reliably yet
        for layer_index, (weights, bias, activation_fun) in enumerate(
                zip(self.weights, self.biases, self.activation_funs)
            ):
            z = matmul_add_bias(current, weights, bias)
            pre_activations.append(z)

            if layer_index == len(self.weights) - 1:
                current = apply_sigmoid(z)
            else:
                current = self.apply_activations(z, layer_index)

            activations.append(current)

        return activations, pre_activations

    def train_batch(self, features, labels):
        batch_size = len(features)
        activations, pre_activations = self.forward(features)
        predictions = activations[-1]

        #print("len", len(predictions), "batch size", batch_size)
        delta = [
            [predictions[row][0] - labels[row]]
            for row in range(batch_size)
        ]

        weight_gradients = [None for _ in self.weights]
        bias_gradients = [None for _ in self.biases]

        for layer_index in reversed(range(len(self.weights))):
            previous_activation = activations[layer_index]
            previous_transposed = transpose(previous_activation)
            grad_w = matmul(previous_transposed, delta)
            grad_b = [0.0 for _ in self.biases[layer_index]]

            for row in delta:
                for col, value in enumerate(row):
                    grad_b[col] += value

            for row in range(len(grad_w)):
                for col in range(len(grad_w[row])):
                    grad_w[row][col] /= batch_size

            for col in range(len(grad_b)):
                grad_b[col] /= batch_size

            weight_gradients[layer_index] = grad_w
            bias_gradients[layer_index] = grad_b

            if layer_index > 0:
                weights_transposed = transpose(self.weights[layer_index])
                previous_delta = matmul(delta, weights_transposed)
                z_previous = pre_activations[layer_index - 1]
                adjusted_delta = zeros_like(previous_delta)

                for row in range(len(previous_delta)):
                    for col in range(len(previous_delta[row])):
                        adjusted_delta[row][col] = (
                            previous_delta[row][col]
                            * self.apply_activation_derivative(
                                z_previous[row][col], layer_index - 1, col
                            )
                        )

                delta = adjusted_delta

        for layer_index in range(len(self.weights)):
            for row in range(len(self.weights[layer_index])):
                for col in range(len(self.weights[layer_index][row])):
                    self.weights[layer_index][row][col] -= (
                        LEARNING_RATE * weight_gradients[layer_index][row][col]
                    )

            for col in range(len(self.biases[layer_index])):
                self.biases[layer_index][col] -= LEARNING_RATE * bias_gradients[layer_index][col]

    def predict_probabilities(self, features):
        activations, _ = self.forward(features)
        return [row[0] for row in activations[-1]]

    def mean_squared_error(self, features, labels):
        probs = self.predict_probabilities(features)
        res = 0
        n = 0
        for i, it in enumerate(probs):
            diff = labels[i] - it
            res += diff * diff
            n += 1
        return res / n


def binary_cross_entropy(predictions, labels):
    epsilon = 1e-12
    loss = 0.0

    for prediction, label in zip(predictions, labels):
        clipped = min(max(prediction, epsilon), 1.0 - epsilon)
        loss += -(label * math.log(clipped) + (1.0 - label) * math.log(1.0 - clipped))

    return loss / len(labels)


def accuracy(predictions, labels):
    correct = 0

    for prediction, label in zip(predictions, labels):
        predicted_label = 1.0 if prediction >= 0.5 else 0.0
        if predicted_label == label:
            correct += 1

    return correct / len(labels)


def write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem",
        "activation",
        "topology",
        "epochs",
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
    ]

    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()

        for row in rows:
            formatted = dict(row)
            for key in [
                "train_loss",
                "test_loss",
                "train_accuracy",
                "test_accuracy",
            ]:
                formatted[key] = f"{row[key]:.6f}"
            writer.writerow(formatted)

def crossover(fa, fb, rng):
    fc = NeuralNetwork(None, None, noinit = True)
    fc.crossover(fa, fb, rng)
    return fc

def tournament(population, batch_features, batch_labels, rng):
    fa = None
    fa_best = math.inf
    for model in rng.choices(population, k = TOURNAMENT_SIZE):
        err = model.mean_squared_error(batch_features, batch_labels)
        if fa == None or err < fa_best:
            fa = model
            fa_best = err
    return fa, fa_best

def select_population(population, batch_features, batch_labels, rng):
    res = []
    while len(res) < POPULATION_SIZE:
        fa, fa_best = tournament(population, batch_features, batch_labels, rng)
        fb, fb_best = tournament(population, batch_features, batch_labels, rng)
        fc = None
        if rng.random() <= CROSSOVER_PROB:
            fc = crossover(fa, fb, rng)
        else:
            fc = fa.copy() if fa_best > fb_best else fb.copy()
        res.append(fc)
    return res

def mutate_population(population, rng):
    for model in population:
        if rng.random() <= MUTATION_PROB:
            for layer in model.activation_funs:
                for i in rng.choices(range(len(layer)), k = math.floor(len(layer) * MUTATION_SELECTION)):
                    layer[i] = get_activation_funs(rng.choice(ActivationList))

def train_population(problem_name, problem_config, rows):
    feature_names, train_features, train_labels = load_dataset(problem_config["train"])
    test_feature_names, test_features, test_labels = load_dataset(problem_config["test"])

    train_features, test_features = standardize(train_features, test_features)
    layer_sizes = [len(feature_names)] + problem_config["hidden_layers"] + [1]
    indices = list(range(len(train_features)))

    population = []
    rng = random.Random(RANDOM_SEED)

    for i in range(POPULATION_SIZE):
        model = NeuralNetwork(layer_sizes, rng)
        population.append(model)

    for _ in range(problem_config["epochs"]):
        rng.shuffle(indices)
        for batch_start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[batch_start : batch_start + BATCH_SIZE]
            batch_features = [train_features[index] for index in batch_indices]
            batch_labels = [train_labels[index] for index in batch_indices]

            population = select_population(population, batch_features, batch_labels, rng)
            mutate_population(population, rng)

            for model in population:
                model.train_batch(batch_features, batch_labels)

    best_loss = math.inf
    best_model = None
    best_predictions = None
    best_i = -1

    for i, model in enumerate(population):
        test_predictions = model.predict_probabilities(test_features)

        test_loss = binary_cross_entropy(test_predictions, test_labels)

        if test_loss < best_loss or best_model == None:
            best_model = model
            best_loss = test_loss
            best_predictions = test_predictions
            best_i = i


    test_predictions = best_predictions
    train_predictions = model.predict_probabilities(train_features)

    test_loss = best_loss
    train_loss = binary_cross_entropy(train_predictions, train_labels)

    train_accuracy = accuracy(train_predictions, train_labels)
    test_accuracy = accuracy(test_predictions, test_labels)

    print(f"{problem_name}: individual #{best_i} train {train_accuracy} test {test_accuracy}")
    rows.append({
        "problem": problem_name,
        "activation": "N/A",
        "topology": "-".join(str(size) for size in layer_sizes),
        "epochs": problem_config["epochs"],
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    })

def main():
    rows = []

    for problem_name, problem_config in PROBLEMS.items():
        train_population(problem_name, problem_config, rows)

    output_path = ROOT / "results/evo_results.csv"
    write_results(output_path, rows)
    print(f"Results written to {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
