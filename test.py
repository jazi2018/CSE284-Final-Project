import argparse
from typing import Optional

import numpy as np
from model import Laihmm


def main(num_admixed_individuals: int = 10, max_ceu_samples: Optional[int] = None):

    # -----------------------------
    # Load genotype dataset
    # -----------------------------
    print("loading ref genotypes (this is a really big file)")
    with np.load("data/chr1_hmm_data.npz") as data:
        ref_genotypes = data["ref_genotypes"]

    # -----------------------------
    # Load allele frequencies
    # -----------------------------
    print("loading allele frequencies")
    freqs = np.genfromtxt(
        "data/chr1_allele_freqs.tsv",
        delimiter="\t",
        names=True,
        dtype=None,
        encoding=None
    )

    ceu_freqs = freqs["freq_CEU"]
    yri_freqs = freqs["freq_YRI"]

    # -----------------------------
    # Load sample metadata
    # -----------------------------
    print("loading population data")
    kg_samples = np.genfromtxt(
        "data/igsr_samples.tsv",
        delimiter="\t",
        names=True,
        dtype=None,
        encoding=None
    )

    # -----------------------------
    # Load reference sample order
    # -----------------------------
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

    # -----------------------------
    # Settings for repeated admixed individuals
    # -----------------------------
    switch_prob = 0.001
    yri_genotype = ref_genotypes[:, yri_samples[0]]

    # -----------------------------
    # Run through CEU reference samples (one-by-one)
    # -----------------------------
    max_ceu = max_ceu_samples or len(ceu_samples)
    for ceu_idx, example_sample in enumerate(ceu_samples[:max_ceu], start=1):
        print(f"\n{'=' * 50}")
        print(f"PROCESSING CEU SAMPLE #{ceu_idx} (ref index {example_sample})")
        print(f"{'=' * 50}")

        # -----------------------------
        # Assign CEU genotype and build emission matrix
        # -----------------------------
        genotype = ref_genotypes[:, example_sample]

        print("generating emission matrix")
        n_snps = genotype.shape[0]

        emission_matrix = np.zeros((2, n_snps))

        mask0 = genotype == 0
        mask1 = genotype == 1
        mask2 = genotype == 2

        # CEU
        emission_matrix[0, mask0] = 1 - ceu_freqs[mask0]
        emission_matrix[0, mask1] = ceu_freqs[mask1]
        emission_matrix[0, mask2] = ceu_freqs[mask2] ** 2

        # YRI
        emission_matrix[1, mask0] = 1 - yri_freqs[mask0]
        emission_matrix[1, mask1] = yri_freqs[mask1]
        emission_matrix[1, mask2] = yri_freqs[mask2] ** 2

        # -----------------------------
        # Run HMM
        # -----------------------------
        print("running HMM")
        model = Laihmm(emission_matrix)
        predictions = model.predict()

        # -----------------------------
        # Evaluate
        # -----------------------------
        print(f"\nRESULTS FOR CEU SAMPLE #{ceu_idx}:")
        predictions = np.array(predictions)

        errors = np.sum(predictions != 0)
        accuracy = (len(predictions) - errors) / len(predictions)

        print(f"accuracy: {accuracy}")
        print(f"number of errors: {errors}\n")

        # -----------------------------
        # Generate and evaluate multiple admixed individuals
        # -----------------------------
        ceu_genotype = genotype
        for indiv_num in range(1, num_admixed_individuals + 1):
            print(f"\n{'=' * 50}")
            print(f"PROCESSING ADMIXED INDIVIDUAL #{indiv_num} (CEU sample #{ceu_idx})")
            print(f"{'=' * 50}")

            # -----------------------------
            # Make fake admixed sample
            # -----------------------------
            print("generating admixed sample")
            n = genotype.shape[0]

            ancestry = np.zeros(n, dtype=int)
            current = np.random.randint(2)

            swaps = 0
            for i in range(n):
                if np.random.rand() < switch_prob:
                    swaps += 1
                    current = 1 - current

                ancestry[i] = current

            print(f"\ttotal swaps: {swaps}")

            admixed = np.where(ancestry == 0, ceu_genotype, yri_genotype)

            # -----------------------------
            # Re-generate emissions matrix
            # -----------------------------
            print("generating emission matrix for admixed individual")
            n_snps = admixed.shape[0]

            emission_matrix = np.zeros((2, n_snps))

            mask0 = admixed == 0
            mask1 = admixed == 1
            mask2 = admixed == 2

            # CEU
            emission_matrix[0, mask0] = 1 - ceu_freqs[mask0]
            emission_matrix[0, mask1] = ceu_freqs[mask1]
            emission_matrix[0, mask2] = ceu_freqs[mask2] ** 2

            # YRI
            emission_matrix[1, mask0] = 1 - yri_freqs[mask0]
            emission_matrix[1, mask1] = yri_freqs[mask1]
            emission_matrix[1, mask2] = yri_freqs[mask2] ** 2

            # -----------------------------
            # Run HMM
            # -----------------------------
            print(f"Running HMM for admixed individual number {indiv_num}")
            model = Laihmm(emission_matrix)
            predictions = model.predict()

            # -----------------------------
            # Evaluate
            # -----------------------------
            print(f"\nRESULTS FOR ADMIXED INDIVIDUAL #{indiv_num} (CEU sample #{ceu_idx}):")
            predictions = np.array(predictions)

            errors = np.sum(predictions != ancestry)
            accuracy = (len(predictions) - errors) / len(predictions)

            print(f"accuracy: {accuracy}")
            print(f"number of errors: {errors}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HMM on reference and simulated admixed individuals."
    )
    parser.add_argument(
        "--num-admixed",
        "-n",
        type=int,
        default=1000,
        help="Number of admixed individuals to generate and evaluate.",
    )
    parser.add_argument(
        "--max-ceu",
        type=int,
        default=None,
        help="Maximum number of CEU reference samples to iterate through (default: all).",
    )

    args = parser.parse_args()
    main(num_admixed_individuals=args.num_admixed, max_ceu_samples=args.max_ceu)