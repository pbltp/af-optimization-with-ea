#!/usr/bin/env python3
# Based on Basic Group's work
# TODO:
# * training accuracy is lower than test accuracy, which is suspicious
# * we still have separate weights, I suppose it is better to have just one?
#   and then just train using the champion (or one at random)
# * save the final distribution of functions too (in a separate file? or as JSON)
# * at the end, retrain best network from scratch
# * maybe try optimizing last layer too (but then we have to add a layer to clamp
#   to 0-1 at the end)
# * check if using delta of loss as fitness is better than just regular loss

import csv
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RANDOM_SEED = 42
LEARNING_RATE = 0.01
BATCH_SIZE = 32
MOMENTUM = 0.1

POPULATION_SIZE = 8
TOURNAMENT_SIZE = 2
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.01


PROBLEMS = {
    "Easy": {
        "train": ROOT / "../problems/easy/two_moons_train.csv",
        "test": ROOT / "../problems/easy/two_moons_test.csv",
        "hidden_layers": [8],
        "epochs": 100,
    },
    "Medium": {
        "train": ROOT / "../problems/medium/concentric_circles_train.csv",
        "test": ROOT / "../problems/medium/concentric_circles_test.csv",
        "hidden_layers": [8, 8],
        "epochs": 150,
    },
    "Hard": {
        "train": ROOT / "../problems/hard/crossing_spirals_train.csv",
        "test": ROOT / "../problems/hard/crossing_spirals_test.csv",
        "hidden_layers": [16, 16],
        "epochs": 250,
    },
}

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        with path.open() as csv_file:
            reader = csv.DictReader(csv_file)
            self.feature_names = [name for name in reader.fieldnames if name != "label"]
            self.features = []
            self.labels = []
            for row in reader:
                tensor = torch.Tensor([float(row[name]) for name in self.feature_names])
                self.features.append(tensor)
                self.labels.append(torch.Tensor([float(row["label"])]))
        feature_count = len(self.features[0])
        means = torch.zeros(feature_count)
        stds = torch.zeros(feature_count)

        # XXX F.normalize?
        for col in range(feature_count):
            values = [row[col] for row in self.features]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(variance)
            means[col] = mean
            stds[col] = std if std > 0 else 1.0

        self.features = [
            torch.Tensor([(row[col] - means[col]) / stds[col] for col in range(feature_count)])
            for row in self.features
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_dataset(path):
    dataset = MyDataset(path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

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

def identity(x):
    return x

# Acceptable activation functions for hidden layers.
ActivationList = [
    torch.relu,         # ReLU
    F.gelu,             # GeLU
    torch.sigmoid,      # sigmoid
    torch.tanh,         # tanh
    F.silu,             # swish
    identity,           # identity
]

class CustomAFNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, rng):
        """
        Initialize a NN with per-neuron custom activation functions.

        @params:
            input_size: int, number of input features.
            layer_sizes: list[int] representing the size of each layer.
        """
        super(CustomAFNeuralNetwork, self).__init__()
        self.layers = []
        self.nets = None
        self.running_loss = 0
        self.prev_loss = 0
        self.loss_delta = 0
        self.last_loss = 0
        self.activation_funs = []
        self.loss_fn = nn.BCELoss()

        for i in range(len(layer_sizes) - 2):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            self.layers.append(nn.Linear(fan_in, fan_out))
            layer_funs = [rng.choice(ActivationList) for _ in range(fan_out)]
            self.activation_funs.append(layer_funs)
            self.layers.append(CustomAFLayer(layer_funs))

        # Final layer: we must use sigmoid, otherwise BCE doesn't really work.
        # (We could try the others + clamp, but I'm not sure if it's worth
        # the trouble.)
        self.layers.append(nn.Linear(layer_sizes[-2], 1))
        self.layers.append(nn.Sigmoid())

        self.nets = nn.Sequential(*self.layers)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    def forward(self, x):
        return self.nets(x)

    def copy_funs(self):
        return copy.deepcopy(self.activation_funs)

class CustomAFLayer(nn.Module):
    """
    Applies a (potentially different) activation function to each neuron column.
    Supports batched input of shape (batch_size, n_neurons).
    """

    def __init__(self, af_conf):
        super().__init__()
        self.af_conf = af_conf

    def forward(self, x):
        af = torch.empty_like(x)

        for i in range(x.shape[-1]):
            col = x[..., i]
            fun = self.af_conf[i]
            af[..., i] = fun(col)

        return af

def crossover(fa, fb, rng):
    res = []
    for layer in range(len(fa.activation_funs)):
        layer_activation = [
            a if rng.random() <= .5 else b for a, b in zip(
                fa.activation_funs[layer], fb.activation_funs[layer]
            )
        ]
        res.append(layer_activation)
    return res

def tournament(population, batch_inputs, batch_labels, rng):
    fa = None
    for model in rng.choices(population, k = TOURNAMENT_SIZE):
        # XXX I'm not sure if delta is better than just regular loss
        if fa == None or model.loss_delta < fa.loss_delta:
            fa = model
    return fa

def select_population(population, batch_inputs, batch_labels, rng):
    res = []
    for i in range(POPULATION_SIZE):
        fa = tournament(population, batch_inputs, batch_labels, rng)
        fb = tournament(population, batch_inputs, batch_labels, rng)
        fc = None
        if rng.random() <= CROSSOVER_PROB:
            fc = crossover(fa, fb, rng)
        else:
            fc = fa.copy_funs() if fa.loss_delta < fb.loss_delta else fb.copy_funs()
        res.append(fc)
    # Redistribute new functions randomly.
    # Note: this is a bit ugly, we're basically modifying the lists the
    # layer objects point to.
    for funs, model in zip(res, rng.sample(population, len(population))):
        for i, layer_funs in enumerate(funs):
            for j, fun in enumerate(layer_funs):
                model.activation_funs[i][j] = fun

def mutate_population(population, rng):
    for model in population:
        for layer in model.activation_funs:
            # XXX we could do sampling more efficiently
            for i in range(len(layer)):
                if rng.random() <= MUTATION_PROB:
                    layer[i] = rng.choice(ActivationList)

def evaluate(model, loader):
    with torch.no_grad():
        size = len(loader.dataset)
        num_batches = len(loader)
        loss = 0
        accuracy = 0

        model.eval()
        for inputs, labels in loader:
            outputs = model(inputs)
            loss_it = model.loss_fn(outputs, labels).item()
            outputs = (outputs>0.5).float()
            accuracy_it = (outputs == labels).float().sum().item()
            loss += loss_it
            accuracy += accuracy_it

        return loss / num_batches, accuracy / size

def train_population(problem_name, problem_config, rows):
    training_loader = load_dataset(problem_config["train"])
    test_loader = load_dataset(problem_config["test"])

    layer_sizes = [len(training_loader.dataset.feature_names)] + problem_config["hidden_layers"] + [1]

    population = []
    rng = random.Random(RANDOM_SEED)

    for i in range(POPULATION_SIZE):
        model = CustomAFNeuralNetwork(layer_sizes, rng)
        model.train()
        population.append(model)

    for num_epoch in range(problem_config["epochs"]):
        # Reset loss for this epoch
        for model in population:
            model.running_loss = 0

        for inputs, labels in training_loader:
            # Evolutionary algorithm
            select_population(population, inputs, labels, rng)
            mutate_population(population, rng)

            # XXX champion only?
            for model in population:
                model.optimizer.zero_grad()
                outputs = model(inputs)
                loss = model.loss_fn(outputs, labels)
                loss.backward()
                # loss - model.prev_loss: negative if previous loss was
                # greater, positive otherwise (smallest wins)
                model.loss_delta = loss - model.prev_loss
                model.prev_loss = loss
                model.optimizer.step() # adjust weights
                model.running_loss += loss.item() # add loss

        print("epoch", num_epoch)
        for i, model in enumerate(population):
            model.last_loss = model.running_loss / BATCH_SIZE # loss per batch
            print("model", i, "loss:", model.last_loss)

    best_loss = math.inf
    best_model = None
    best_accuracy = 0
    best_i = -1

    for i, model in enumerate(population):
        loss, accuracy = evaluate(model, test_loader)

        # XXX output all models sorted instead?
        if loss < best_loss or best_model == None:
            best_model = model
            best_loss = loss
            best_accuracy = accuracy
            best_i = i

    test_loss, test_accuracy = best_loss, best_accuracy
    train_loss, train_accuracy = evaluate(model, training_loader)

    print(f"{problem_name}: individual #{best_i} train {train_accuracy} test {test_accuracy} train loss {train_loss} train loss really? {population[best_i].last_loss} test loss {test_loss}")

    print("\n".join(
        ", ".join(fun.__name__ for fun in funs) for funs in model.activation_funs)
    )
    rows.append({
        "problem": problem_name,
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

    torch.manual_seed(RANDOM_SEED)

    output_path = ROOT / "../results/evo_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem",
        "topology",
        "epochs",
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
    ]

    with output_path.open("w", newline="") as csv_file:
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

    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
