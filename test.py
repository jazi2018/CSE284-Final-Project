import argparse
from typing import Optional

import numpy as np
from model import Laihmm


def build_emission_matrix(
    genotype: np.ndarray,
    ceu_freqs: np.ndarray,
    yri_freqs: np.ndarray
) -> np.ndarray:
    """
    Build a 2 x n_snps emission matrix for the given genotype vector.
    Row 0 = CEU emissions
    Row 1 = YRI emissions
    """
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

    return emission_matrix


def main(
    num_admixed_individuals: int = 10,
    max_ceu_samples: Optional[int] = None,
    yri_sample_index: Optional[int] = None
):
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

    if len(ceu_samples) == 0:
        raise ValueError("No CEU samples found.")
    if len(yri_samples) == 0:
        raise ValueError("No YRI samples found.")

    if yri_sample_index is not None:
        if yri_sample_index < 0 or yri_sample_index >= len(yri_samples):
            raise ValueError(
                f"yri_sample_index must be between 0 and {len(yri_samples) - 1}, "
                f"got {yri_sample_index}."
            )

    # -----------------------------
    # Settings for repeated admixed individuals
    # -----------------------------
    switch_prob = 0.001

    # -----------------------------
    # Run through CEU reference samples (one-by-one)
    # -----------------------------
    max_ceu = len(ceu_samples) if max_ceu_samples is None else max_ceu_samples

    for ceu_idx, example_sample in enumerate(ceu_samples[:max_ceu], start=1):
        print(f"\n{'=' * 50}")
        print(f"PROCESSING CEU SAMPLE #{ceu_idx} (ref index {example_sample})")
        print(f"{'=' * 50}")

        # -----------------------------
        # Assign CEU genotype and build emission matrix
        # -----------------------------
        genotype = ref_genotypes[:, example_sample]

        print("generating emission matrix")
        emission_matrix = build_emission_matrix(genotype, ceu_freqs, yri_freqs)

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
            # Select YRI sample
            # -----------------------------
            if yri_sample_index is None:
                chosen_yri_pos = np.random.randint(len(yri_samples))
                print(
                    f"randomly selected YRI sample position {chosen_yri_pos} "
                    f"(ref index {yri_samples[chosen_yri_pos]})"
                )
            else:
                chosen_yri_pos = yri_sample_index
                print(
                    f"using fixed YRI sample position {chosen_yri_pos} "
                    f"(ref index {yri_samples[chosen_yri_pos]})"
                )

            yri_genotype = ref_genotypes[:, yri_samples[chosen_yri_pos]]

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
            # Build emission matrix
            # -----------------------------
            print("generating emission matrix for admixed individual")
            emission_matrix = build_emission_matrix(admixed, ceu_freqs, yri_freqs)

            # -----------------------------
            # Run HMM
            # -----------------------------
            print(f"running HMM for admixed individual number {indiv_num}")
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
        help="Number of admixed individuals to generate and evaluate per CEU sample.",
    )
    parser.add_argument(
        "--max-ceu",
        type=int,
        default=None,
        help="Maximum number of CEU reference samples to iterate through (default: all).",
    )
    parser.add_argument(
        "--yri-sample-index",
        type=int,
        default=None,
        help=(
            "0-based index into the YRI sample list to force a fixed YRI sample. "
            "If omitted, a random YRI sample is chosen for each admixed individual."
        ),
    )

    args = parser.parse_args()
    main(
        num_admixed_individuals=args.num_admixed,
        max_ceu_samples=args.max_ceu,
        yri_sample_index=args.yri_sample_index
    )