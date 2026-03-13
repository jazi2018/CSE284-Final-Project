import numpy as np
from model import Laihmm, BetterStatesLaihmm

"""
This script runs tests on both the Laihmm and BetterStatesLaihmm models.
It's contents are effectively a reimplementation of the unified test in the notebook,
but doesn't require ipykernel or matplotlib to be installed.
"""

def load_reference_data():
    print("loading ref genotypes (this is a really big file)")
    with np.load("data/chr1_hmm_data_phased.npz") as data:
        ref_haplotypes = data["ref_hap0"]

    print("loading allele frequencies")
    freqs = np.genfromtxt(
        "data/chr1_allele_freqs.tsv",
        delimiter="\t",
        names=True,
        dtype=None,
        encoding=None,
    )

    ceu_freqs = freqs["freq_CEU"]
    yri_freqs = freqs["freq_YRI"]

    print("loading population data")
    kg_samples = np.genfromtxt(
        "data/igsr_samples.tsv",
        delimiter="\t",
        names=True,
        dtype=None,
        encoding=None,
    )

    print("separating ceu and yri samples")
    with open("data/ref_samples.txt") as f:
        samples = [line.strip() for line in f]

    sample_to_pop = dict(
        zip(kg_samples["Sample_name"], kg_samples["Population_code"])
    )

    ceu_samples = []
    yri_samples = []

    for i, sample in enumerate(samples):
        code = sample_to_pop.get(sample)
        if code == "CEU":
            ceu_samples.append(i)
        elif code == "YRI":
            yri_samples.append(i)

    return ref_haplotypes, ceu_freqs, yri_freqs, np.array(ceu_samples), np.array(
        yri_samples
    )


def build_trivial_emission(ref_haplotypes, ceu_freqs, yri_freqs, ceu_samples):
    print("assigning example sample - first CEU index")
    example_sample = int(ceu_samples[0])
    haplotype = ref_haplotypes[:, example_sample]

    print("generating emission matrix")
    n_snps = haplotype.shape[0]
    emission_matrix_trivial = np.zeros((2, n_snps))

    mask0 = haplotype == 0
    mask1 = haplotype == 1

    emission_matrix_trivial[0, mask0] = 1 - ceu_freqs[mask0]
    emission_matrix_trivial[0, mask1] = ceu_freqs[mask1]

    emission_matrix_trivial[1, mask0] = 1 - yri_freqs[mask0]
    emission_matrix_trivial[1, mask1] = yri_freqs[mask1]

    return haplotype, emission_matrix_trivial


def load_query_data(ceu_freqs, yri_freqs):
    print("loading query data")
    with np.load("data/chr1_hmm_data_phased.npz") as data:
        query_haplotypes = data["query_hap0"]
        positions = data["positions"]

    print("assigning query sample - first index")
    haplotype = query_haplotypes[:, 0]

    print("generating emission matrix")
    n_snps = haplotype.shape[0]
    emission_matrix = np.zeros((2, n_snps))

    mask0 = haplotype == 0
    mask1 = haplotype == 1

    emission_matrix[0, mask0] = 1 - ceu_freqs[mask0]
    emission_matrix[0, mask1] = ceu_freqs[mask1]

    emission_matrix[1, mask0] = 1 - yri_freqs[mask0]
    emission_matrix[1, mask1] = yri_freqs[mask1]

    print("generating ground truth for query sample")
    recomb_positions = []
    with open("data/sim_admixed_chr1.bp") as f:
        f.readline()
        for line in f:
            if len(line.split()) < 4:
                break
            line = line.split()
            recomb_positions.append((line[0], int(line[2])))

    ancestry_map = {"CEU": 0, "YRI": 1}
    bps = np.array([p for _, p in recomb_positions])
    labels = np.array([ancestry_map[a] for a, _ in recomb_positions])
    segment_indices = np.searchsorted(bps, positions, side="right")
    ground_truth = labels[segment_indices]
    assert ground_truth.shape == (n_snps,)

    return haplotype, emission_matrix, ground_truth, positions


def run_laihmm_trivial(emission_matrix_trivial):
    print("running HMM")
    model = Laihmm(emission_matrix_trivial, transition_prob=0.001)
    predictions = np.array(model.predict())

    print('\nRESULTS FOR "100%" CEU BACKGROUND:')
    errors = np.sum(predictions != 0)
    accuracy = (len(predictions) - errors) / len(predictions)
    print(f"accuracy: {accuracy}")
    print(f"number of errors: {errors}")


def run_laihmm_admixed(emission_matrix, ground_truth):
    print("running HMM for admixed individual")
    model = Laihmm(emission_matrix, transition_prob=0.001)
    predictions = np.array(model.predict())

    print("\nRESULTS FOR ADMIXED:")
    errors = np.sum(predictions != ground_truth)
    accuracy = (len(predictions) - errors) / len(predictions)
    print(f"accuracy: {accuracy}")
    print(f"number of errors: {errors}")


def build_reference_panel(ref_haplotypes, ceu_samples, yri_samples):
    print("getting reference panel for better states model")
    ref_indices = np.concatenate((ceu_samples, yri_samples))
    panel = ref_haplotypes[:, ref_indices].T
    ancestries = np.concatenate(
        (np.full(len(ceu_samples), 0), np.full(len(yri_samples), 1))
    )
    return panel, ancestries


def run_better_states(panel, ancestries, haplotype, ground_truth):
    print("running BetterStatesLaihmm on admixed target")
    model = BetterStatesLaihmm(panel, ancestries)
    ancestry_sequence = np.array(model.predict(haplotype))

    print("\nRESULTS FOR ADMIXED WITH BETTER STATES MODEL:")
    errors = np.sum(ancestry_sequence != ground_truth)
    accuracy = (len(ancestry_sequence) - errors) / len(ancestry_sequence)
    print(f"accuracy: {accuracy}")
    print(f"number of errors: {errors}")


def main():
    print("Unified test: Laihmm + BetterStatesLaihmm")
    print("Assuming working directory is the repository root.\n")

    (
        ref_haplotypes,
        ceu_freqs,
        yri_freqs,
        ceu_samples,
        yri_samples,
    ) = load_reference_data()

    trivial_haplotype, emission_matrix_trivial = build_trivial_emission(
        ref_haplotypes, ceu_freqs, yri_freqs, ceu_samples
    )

    query_haplotype, emission_matrix, ground_truth, positions = load_query_data(
        ceu_freqs, yri_freqs
    )

    run_laihmm_trivial(emission_matrix_trivial)
    run_laihmm_admixed(emission_matrix, ground_truth)

    panel, ancestries = build_reference_panel(
        ref_haplotypes, ceu_samples, yri_samples
    )
    run_better_states(panel, ancestries, query_haplotype, ground_truth)


if __name__ == "__main__":
    main()

