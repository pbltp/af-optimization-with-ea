import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# =====================================================
# Style
# =====================================================

sns.set_theme(
    style="whitegrid",
    context="talk"
)

RESULTS_DIR = Path("results")

POPULATION_TICKS = [
    100,
    200,
    400,
    800,
    1600,
    3200
]

# =====================================================
# CSV Dateien laden
# =====================================================


def load_results():

    all_data = []

    for file in RESULTS_DIR.glob("*.csv"):

        name = file.stem.lower()

        pop_match = re.search(r'(\d+)', name)

        if not pop_match:
            continue

        population = int(pop_match.group(1))

        mutation = (
            "Mutation"
            if "_m_" in name
            else "No Mutation"
        )

        df = pd.read_csv(file)

        df["population"] = population
        df["mutation"] = mutation

        all_data.append(df)

        print(
            f"Loaded {file.name} "
            f"(Population={population}, {mutation})"
        )

    if not all_data:
        raise RuntimeError(
            "Keine CSV-Dateien gefunden."
        )

    return pd.concat(
        all_data,
        ignore_index=True
    )


# =====================================================
# Nur Evolutionszeilen
# =====================================================

def get_evolution_rows(df):

    return df[
        df["functions"]
        ==
        "evolution result (functions not fixed)"
    ].copy()


# =====================================================
# 1) Population Scaling
# =====================================================

def plot_population_scaling(df):

    evo = get_evolution_rows(df)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        sharey=True
    )

    for ax, mutation in zip(
        axes,
        ["No Mutation", "Mutation"]
    ):

        subset = evo[
            evo["mutation"] == mutation
        ]

        sns.lineplot(
            data=subset,
            x="population",
            y="test_accuracy",
            hue="problem",
            marker="o",
            linewidth=3,
            markersize=10,
            ax=ax
        )

        ax.set_xscale("log", base=2)

        ax.set_xticks(POPULATION_TICKS)
        ax.set_xticklabels(
            [str(x) for x in POPULATION_TICKS]
        )

        ax.set_ylim(0, 1)

        ax.set_title(
            mutation,
            fontsize=18,
            weight="bold"
        )

        ax.set_xlabel(
            "Population Size"
        )

        ax.set_ylabel(
            "Test Accuracy"
        )

        ax.grid(
            alpha=0.3
        )

    fig.suptitle(
        "Influence of Population Size on Test Accuracy",
        fontsize=22,
        weight="bold"
    )

    plt.tight_layout()

    outfile = (
        RESULTS_DIR
        / "population_scaling.png"
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {outfile}")


# =====================================================
# 2) Mutation vs No Mutation
# =====================================================

def plot_mutation_comparison(df):

    evo = get_evolution_rows(df)

    best_rows = []

    for problem in sorted(
        evo["problem"].unique()
    ):

        for mutation in [
            "No Mutation",
            "Mutation"
        ]:

            subset = evo[
                (evo["problem"] == problem)
                &
                (evo["mutation"] == mutation)
            ]

            if len(subset) == 0:
                continue

            best = subset.loc[
                subset["test_accuracy"].idxmax()
            ]

            best_rows.append(best)

    best_df = pd.DataFrame(best_rows)

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=best_df,
        x="problem",
        y="test_accuracy",
        hue="mutation"
    )

    plt.ylim(0, 1)

    plt.title(
        "Best Accuracy: Retraining",
        fontsize=20,
        weight="bold"
    )

    plt.ylabel(
        "Best Test Accuracy"
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.3f"
        )

    plt.tight_layout()

    outfile = (
        RESULTS_DIR
        / "mutation_vs_nomutation.png"
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {outfile}")


# =====================================================
# 3) Retraining Validation
# =====================================================

def plot_retraining_validation(df):

    evolution = df[
        df["functions"]
        ==
        "evolution result (functions not fixed)"
    ].copy()

    retrained = df[
        df["functions"]
        !=
        "evolution result (functions not fixed)"
    ].copy()

    rows = []

    for problem in sorted(
        evolution["problem"].unique()
    ):

        best_evo = evolution[
            evolution["problem"] == problem
        ].sort_values(
            "test_accuracy",
            ascending=False
        ).iloc[0]

        match = retrained[
            (retrained["problem"]
             == problem)
            &
            (retrained["population"]
             == best_evo["population"])
            &
            (retrained["mutation"]
             == best_evo["mutation"])
        ]

        if len(match) == 0:
            continue

        best_retrain = match.iloc[0]

        rows.append({
            "problem": problem,
            "phase": "Evolution",
            "accuracy": best_evo["test_accuracy"]
        })

        rows.append({
            "problem": problem,
            "phase": "Retraining",
            "accuracy": best_retrain["test_accuracy"]
        })

    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=plot_df,
        x="problem",
        y="accuracy",
        hue="phase"
    )

    plt.ylim(0, 1)

    plt.title(
        "Retraining Validation",
        fontsize=20,
        weight="bold"
    )

    plt.ylabel(
        "Test Accuracy"
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.3f"
        )

    plt.tight_layout()

    outfile = (
        RESULTS_DIR
        / "retraining_validation.png"
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {outfile}")


# =====================================================
# 4) Evolution vs Basic Activations
# =====================================================

def plot_evolution_vs_baseline(df):

    baseline = pd.read_csv(
        RESULTS_DIR / "base_result.csv"
    )

    evo = get_evolution_rows(df)

    rows = []

    for problem in sorted(
        baseline["problem"].unique()
    ):

        # bestes Baseline-Modell
        best_base = baseline[
            baseline["problem"] == problem
        ].sort_values(
            "test_accuracy",
            ascending=False
        ).iloc[0]

        # bestes Evolutionsergebnis
        best_evo = evo[
            evo["problem"] == problem
        ].sort_values(
            "test_accuracy",
            ascending=False
        ).iloc[0]

        rows.append({
            "problem": problem,
            "method": "Best Baseline",
            "accuracy": best_base["test_accuracy"]
        })

        rows.append({
            "problem": problem,
            "method": "Evolution",
            "accuracy": best_evo["test_accuracy"]
        })

    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=plot_df,
        x="problem",
        y="accuracy",
        hue="method"
    )

    plt.ylim(0, 1)

    plt.title(
        "Evolution vs Best Base",
        fontsize=20,
        weight="bold"
    )

    plt.ylabel(
        "Test Accuracy"
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.3f"
        )

    plt.tight_layout()

    outfile = (
        RESULTS_DIR
        / "evolution_vs_baseline.png"
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {outfile}")

# =====================================================
# Zusammenfassung
# =====================================================


def print_summary(df):

    evo = get_evolution_rows(df)

    print("\n")
    print("=" * 70)
    print("BEST RESULTS")
    print("=" * 70)

    for problem in sorted(
        evo["problem"].unique()
    ):

        best = evo[
            evo["problem"] == problem
        ].sort_values(
            "test_accuracy",
            ascending=False
        ).iloc[0]

        print(
            f"\n{problem}"
        )

        print(
            f"Population : {best['population']}"
        )

        print(
            f"Mutation   : {best['mutation']}"
        )

        print(
            f"Accuracy   : "
            f"{best['test_accuracy']:.4f}"
        )

        print(
            f"Loss       : "
            f"{best['test_loss']:.4f}"
        )


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    print(
        "\nLoading experiment results...\n"
    )

    df = load_results()

    print(
        f"\nTotal rows loaded: {len(df)}"
    )

    plot_population_scaling(df)

    plot_mutation_comparison(df)

    plot_retraining_validation(df)

    plot_evolution_vs_baseline(df)

    print_summary(df)

    print(
        "\nDone.\n"
    )
