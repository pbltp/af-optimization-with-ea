#!/usr/bin/env python3
# Based on Basic Group's work
# TODO:
# * maybe try optimizing last layer too? (but then we have to add a layer
#   to clamp to 0-1 at the end)

import csv
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Same as basics group.
RANDOM_SEED = 42
LEARNING_RATE = 0.01
BATCH_SIZE = 32
MOMENTUM = 0

# Note: a greater population size indicates that the network is trained
# more often on the same batch.
POPULATION_SIZE = 3200
TOURNAMENT_SIZE = 2
CROSSOVER_PROB = 0.8

# Same as basics group.
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
            self.feature_names = [
                name for name in reader.fieldnames if name != "label"]
            self.features = []
            self.labels = []
            for row in reader:
                tensor = torch.Tensor([float(row[name])
                                      for name in self.feature_names])
                self.features.append(tensor)
                self.labels.append(torch.Tensor([float(row["label"])]))
        feature_count = len(self.features[0])
        means = torch.zeros(feature_count)
        stds = torch.zeros(feature_count)

        # XXX F.normalize?
        for col in range(feature_count):
            values = [row[col] for row in self.features]
            mean = sum(values) / len(values)
            variance = sum((value - mean) **
                           2 for value in values) / len(values)
            std = math.sqrt(variance)
            means[col] = mean
            stds[col] = std if std > 0 else 1.0

        self.features = [
            torch.Tensor([(row[col] - means[col]) / stds[col]
                         for col in range(feature_count)])
            for row in self.features
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_dataset(path):
    dataset = MyDataset(path)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True)
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
            [(row[col] - means[col]) / stds[col]
             for col in range(feature_count)]
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


class CustomAFList():
    def __init__(self, layer_sizes, rng, funs=None):
        """
        Initialize a custom AF list with random functions.

        @params:
            fan_out: list of output neurons.
            rng: random.Random a random number generator.
        """

        self.current_loss = 0
        if funs != None:
            self.funs = funs
        else:
            self.funs = []

            for i in range(len(layer_sizes) - 2):
                fan_out = layer_sizes[i + 1]
                layer_funs = [rng.choice(ActivationList)
                              for _ in range(fan_out)]
                self.funs.append(layer_funs)

    def copy(self):
        return CustomAFList(None, None, self.funs)


class CustomAFNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, rng):
        """
        Initialize a NN with layers that accept per-neuron custom activation
        functions.

        @params:
            layer_sizes: list[int] representing the size of each layer.
        """
        super(CustomAFNeuralNetwork, self).__init__()
        self.nets = None
        self.loss_fn = nn.BCELoss()
        self.af_layers = []
        self.af_list = None

        layers = []

        for i in range(len(layer_sizes) - 2):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            layers.append(nn.Linear(fan_in, fan_out))
            layer_funs = [rng.choice(ActivationList) for _ in range(fan_out)]
            af_layer = CustomAFLayer()
            layers.append(af_layer)
            self.af_layers.append(af_layer)

        # Final layer: we must use sigmoid, otherwise binary cross entropy
        # doesn't work.  (We could try the others + clamp, I suppose.)
        layers.append(nn.Linear(layer_sizes[-2], 1))
        layers.append(nn.Sigmoid())

        self.nets = nn.Sequential(*layers)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    def forward(self, x):
        return self.nets(x)

    def set_af_list(self, af_list):
        self.af_list = af_list
        for i, layer in enumerate(af_list.funs):
            self.af_layers[i].af_conf = layer


class CustomAFLayer(nn.Module):
    """
    Applies a (potentially different) activation function to each neuron column.
    Supports batched input of shape (batch_size, n_neurons).
    """

    def __init__(self):
        super().__init__()
        self.af_conf = None

    def forward(self, x):
        af = torch.empty_like(x)

        for i in range(x.shape[-1]):
            col = x[..., i]
            fun = self.af_conf[i]
            af[..., i] = fun(col)

        return af


def crossover(fa, fb, rng):
    res = []
    for layer in range(len(fa.funs)):
        layer_activation = [
            a if rng.random() <= .5 else b for a, b in zip(
                fa.funs[layer], fb.funs[layer]
            )
        ]
        res.append(layer_activation)
    return CustomAFList(None, None, res)


def take_best(fa, fb):
    return fa if fa.current_loss < fb.current_loss else fb


def tournament(model, population, inputs, labels, rng):
    fa = None
    for af_list in rng.choices(population, k=TOURNAMENT_SIZE):
        model.set_af_list(af_list)
        outputs = model(inputs)
        af_list.current_loss = model.loss_fn(outputs, labels).item()
        if fa == None or af_list.current_loss < fa.current_loss:
            fa = af_list
    return fa


def select_population(model, population, inputs, labels, rng):
    res = []
    model.eval()
    with torch.no_grad():
        for i in range(POPULATION_SIZE):
            fa = tournament(model, population, inputs, labels, rng)
            fb = tournament(model, population, inputs, labels, rng)
            fc = None
            if rng.random() <= CROSSOVER_PROB:
                fc = crossover(fa, fb, rng)
            else:
                fc = take_best(fa, fb)
            res.append(fc)
    return res


def mutate_population(population, rng):
    for af_list in population:
        for layer in af_list.funs:
            for i in range(len(layer)):
                if rng.random() <= MUTATION_PROB:
                    layer[i] = rng.choice(ActivationList)


def train_model(model, af_list, inputs, labels):
    model.train()
    model.optimizer.zero_grad()
    model.set_af_list(af_list)
    outputs = model(inputs)
    loss = model.loss_fn(outputs, labels)
    loss.backward()
    model.optimizer.step()  # adjust weights
    af_list.current_loss = loss.item()


def evaluate(model, af_list, loader):
    with torch.no_grad():
        size = len(loader.dataset)
        num_batches = len(loader)
        loss = 0
        accuracy = 0

        model.eval()
        for inputs, labels in loader:
            outputs = model(inputs)
            loss_it = model.loss_fn(outputs, labels).item()
            outputs = (outputs > 0.5).float()
            accuracy_it = (outputs == labels).float().sum().item()
            loss += loss_it
            accuracy += accuracy_it

        return loss / num_batches, accuracy / size


def train_population(problem_name, problem_config, rows):
    training_loader = load_dataset(problem_config["train"])
    test_loader = load_dataset(problem_config["test"])

    layer_sizes = [len(training_loader.dataset.feature_names)
                   ] + problem_config["hidden_layers"] + [1]

    population = []
    rng = random.Random(RANDOM_SEED)

    for i in range(POPULATION_SIZE):
        population.append(CustomAFList(layer_sizes, rng))

    print(problem_name + ": run evolution")
    model = CustomAFNeuralNetwork(layer_sizes, rng)
    num_train = 0
    num_batch = 0
    for num_epoch in range(problem_config["epochs"]):
        for inputs, labels in training_loader:
            if num_batch < num_train:
                # Ensure that the model is trained for as much batches
                # during evolution as during retraining.
                # (Another way would be to divide epochs by population, but
                # this seems more accurate.)
                num_batch += 1
                continue

            # Evolutionary algorithm
            population = select_population(
                model, population, inputs, labels, rng)
            mutate_population(population, rng)

            for af_list in population:
                train_model(model, af_list, inputs, labels)
                num_train += 1
            num_batch += 1

        if num_epoch % 10 == 9:
            print("\rEvolution epoch", num_epoch, end='', flush=True)
    print("")
    print("Trained for", num_train, "batches")

    best_loss = math.inf
    best_af_list = None
    best_i = -1

    for i, af_list in enumerate(population):
        loss, accuracy = evaluate(model, af_list, test_loader)
        if loss < best_loss or best_af_list == None:
            best_af_list = af_list
            best_loss = loss
            best_i = i
    af_list = best_af_list

    test_loss, test_accuracy = evaluate(model, af_list, test_loader)
    train_loss, train_accuracy = evaluate(model, af_list, training_loader)

    rows.append({
        "problem": problem_name,
        "topology": "-".join(str(size) for size in layer_sizes),
        "epochs": problem_config["epochs"],
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "functions": "evolution result (functions not fixed)"
    })

    # Redo training from scratch with the final activation function list.
    # This is to ensure that the benchmark is comparable to the basic case.
    model = CustomAFNeuralNetwork(layer_sizes, rng)

    num_train = 0
    for num_epoch in range(problem_config["epochs"]):
        running_loss = 0
        for inputs, labels in training_loader:
            # Evolutionary algorithm
            train_model(model, af_list, inputs, labels)
            running_loss += af_list.current_loss  # add loss
            num_train += 1
        last_loss = running_loss / BATCH_SIZE  # loss per batch
        if num_epoch % 10 == 9:
            print("\rRetrain epoch", num_epoch, "loss:",
                  last_loss, end='', flush=True)
    print("")
    print("Trained for", num_train, "batches")

    test_loss, test_accuracy = evaluate(model, af_list, test_loader)
    train_loss, train_accuracy = evaluate(model, af_list, training_loader)

    print(f"{problem_name}: individual #{best_i} train {train_accuracy} test {test_accuracy} train loss {train_loss} test loss {test_loss}")

    functions = "|".join(
        ":".join(fun.__name__ for fun in funs) for funs in best_af_list.funs
    )

    print(functions)

    rows.append({
        "problem": problem_name,
        "topology": "-".join(str(size) for size in layer_sizes),
        "epochs": problem_config["epochs"],
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "functions": functions
    })


def main():
    rows = []

    for problem_name, problem_config in PROBLEMS.items():
        train_population(problem_name, problem_config, rows)

    torch.manual_seed(RANDOM_SEED)

    output_path = ROOT / "../results/evo_results_3200_m_001.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem",
        "topology",
        "epochs",
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
        "functions",
    ]

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, lineterminator="\n")
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
