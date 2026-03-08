import argparse
import csv
from pathlib import Path
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


def ensure_max_ceu(max_ceu_samples: Optional[int], n_ceu_available: int) -> int:
    """
    Resolve max_ceu_samples into a valid number of CEU samples to process.
    """
    if max_ceu_samples is None:
        return n_ceu_available

    if max_ceu_samples < 1:
        raise ValueError(f"max_ceu_samples must be at least 1, got {max_ceu_samples}.")

    return min(max_ceu_samples, n_ceu_available)


def get_posteriors_if_available(model: Laihmm) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Best-effort helper to extract CEU/YRI posterior probabilities if the model
    implementation exposes them under a known method/attribute name.

    Returns
    -------
    p_ceu, p_yri : Optional[np.ndarray], Optional[np.ndarray]
        Arrays of length n_snps if available, otherwise (None, None).
    """
    candidate_methods = [
        "posterior_probs",
        "predict_proba",
        "posteriors",
        "forward_backward",
    ]
    candidate_attributes = [
        "posterior_probs",
        "posteriors",
        "gamma",
    ]

    arr = None

    for method_name in candidate_methods:
        if hasattr(model, method_name):
            maybe_method = getattr(model, method_name)
            if callable(maybe_method):
                try:
                    arr = maybe_method()
                    break
                except TypeError:
                    pass

    if arr is None:
        for attr_name in candidate_attributes:
            if hasattr(model, attr_name):
                arr = getattr(model, attr_name)
                break

    if arr is None:
        return None, None

    arr = np.asarray(arr)

    # Expected shape: (2, n_snps) or (n_snps, 2)
    if arr.ndim != 2 or 2 not in arr.shape:
        return None, None

    if arr.shape[0] == 2:
        p_ceu = arr[0, :]
        p_yri = arr[1, :]
    else:
        p_ceu = arr[:, 0]
        p_yri = arr[:, 1]

    return np.asarray(p_ceu), np.asarray(p_yri)


def main(
    num_admixed_individuals: int = 10,
    max_ceu_samples: Optional[int] = None,
    yri_sample_index: Optional[int] = None,
    write_output: bool = False,
    output_dir: str = "output"
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
    # Determine how many CEU samples will be processed
    # -----------------------------
    max_ceu = ensure_max_ceu(max_ceu_samples, len(ceu_samples))

    # -----------------------------
    # Prepare optional CSV writers
    # -----------------------------
    summary_writer = None
    locus_writer = None
    summary_file = None
    locus_file = None

    if write_output:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_path = out_dir / (
            f"lai_summary_ceu{max_ceu}_admixed{num_admixed_individuals}.csv"
        )
        locus_path = out_dir / (
            f"lai_locus_predictions_ceu{max_ceu}_admixed{num_admixed_individuals}.csv"
        )

        print(f"writing summary output to: {summary_path}")
        print(f"writing locus-level output to: {locus_path}")

        summary_file = open(summary_path, "w", newline="")
        locus_file = open(locus_path, "w", newline="")

        summary_writer = csv.writer(summary_file)
        locus_writer = csv.writer(locus_file)

        summary_writer.writerow([
            "run_type",
            "ceu_sample_num",
            "ceu_ref_index",
            "yri_sample_pos",
            "yri_ref_index",
            "admixed_indiv_num",
            "switch_prob",
            "n_snps",
            "num_switches",
            "accuracy",
            "num_errors",
            "mean_true_ancestry",
            "mean_predicted_ancestry",
            "r2_hardcall"
        ])

        locus_writer.writerow([
            "run_type",
            "ceu_sample_num",
            "ceu_ref_index",
            "yri_sample_pos",
            "yri_ref_index",
            "admixed_indiv_num",
            "snp_index",
            "true_ancestry",
            "predicted_ancestry",
            "p_ceu",
            "p_yri"
        ])

    try:
        # -----------------------------
        # Run through CEU reference samples (one-by-one)
        # -----------------------------
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
            # Run HMM on pure CEU
            # -----------------------------
            print("running HMM")
            model = Laihmm(emission_matrix)
            predictions = np.asarray(model.predict())
            p_ceu, p_yri = get_posteriors_if_available(model)

            # -----------------------------
            # Evaluate pure CEU
            # -----------------------------
            print(f"\nRESULTS FOR CEU SAMPLE #{ceu_idx}:")
            true_ancestry = np.zeros_like(predictions, dtype=int)

            errors = np.sum(predictions != true_ancestry)
            accuracy = (len(predictions) - errors) / len(predictions)

            print(f"accuracy: {accuracy}")
            print(f"number of errors: {errors}\n")

            mean_true = float(np.mean(true_ancestry))
            mean_pred = float(np.mean(predictions))

            if np.std(true_ancestry) == 0 or np.std(predictions) == 0:
                r2_hardcall = np.nan
            else:
                r2_hardcall = float(np.corrcoef(true_ancestry, predictions)[0, 1] ** 2)

            if summary_writer is not None:
                summary_writer.writerow([
                    "reference",
                    ceu_idx,
                    example_sample,
                    "",
                    "",
                    "",
                    switch_prob,
                    len(predictions),
                    0,
                    accuracy,
                    int(errors),
                    mean_true,
                    mean_pred,
                    r2_hardcall
                ])

            if locus_writer is not None:
                for snp_idx in range(len(predictions)):
                    locus_writer.writerow([
                        "reference",
                        ceu_idx,
                        example_sample,
                        "",
                        "",
                        "",
                        snp_idx,
                        int(true_ancestry[snp_idx]),
                        int(predictions[snp_idx]),
                        "" if p_ceu is None else float(p_ceu[snp_idx]),
                        "" if p_yri is None else float(p_yri[snp_idx])
                    ])

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

                chosen_yri_ref_index = yri_samples[chosen_yri_pos]
                yri_genotype = ref_genotypes[:, chosen_yri_ref_index]

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
                predictions = np.asarray(model.predict())
                p_ceu, p_yri = get_posteriors_if_available(model)

                # -----------------------------
                # Evaluate
                # -----------------------------
                print(f"\nRESULTS FOR ADMIXED INDIVIDUAL #{indiv_num} (CEU sample #{ceu_idx}):")

                errors = np.sum(predictions != ancestry)
                accuracy = (len(predictions) - errors) / len(predictions)

                print(f"accuracy: {accuracy}")
                print(f"number of errors: {errors}")

                mean_true = float(np.mean(ancestry))
                mean_pred = float(np.mean(predictions))

                if np.std(ancestry) == 0 or np.std(predictions) == 0:
                    r2_hardcall = np.nan
                else:
                    r2_hardcall = float(np.corrcoef(ancestry, predictions)[0, 1] ** 2)

                if summary_writer is not None:
                    summary_writer.writerow([
                        "admixed",
                        ceu_idx,
                        example_sample,
                        chosen_yri_pos,
                        chosen_yri_ref_index,
                        indiv_num,
                        switch_prob,
                        len(predictions),
                        swaps,
                        accuracy,
                        int(errors),
                        mean_true,
                        mean_pred,
                        r2_hardcall
                    ])

                if locus_writer is not None:
                    for snp_idx in range(len(predictions)):
                        locus_writer.writerow([
                            "admixed",
                            ceu_idx,
                            example_sample,
                            chosen_yri_pos,
                            chosen_yri_ref_index,
                            indiv_num,
                            snp_idx,
                            int(ancestry[snp_idx]),
                            int(predictions[snp_idx]),
                            "" if p_ceu is None else float(p_ceu[snp_idx]),
                            "" if p_yri is None else float(p_yri[snp_idx])
                        ])

    finally:
        if summary_file is not None:
            summary_file.close()
        if locus_file is not None:
            locus_file.close()


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
    parser.add_argument(
        "--write-output",
        action="store_true",
        help=(
            "If set, write summary and locus-level CSV outputs. "
            "Default behavior is to not write files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to write output CSV files into if --write-output is set.",
    )

    args = parser.parse_args()
    main(
        num_admixed_individuals=args.num_admixed,
        max_ceu_samples=args.max_ceu,
        yri_sample_index=args.yri_sample_index,
        write_output=args.write_output,
        output_dir=args.output_dir
    )