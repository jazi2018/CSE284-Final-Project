import numpy as np

from model import BetterStatesLaihmm


def main():
    # -----------------------------
    # Load phased reference panel
    # -----------------------------
    print("loading phased reference panel for chr22")
    with np.load("data/chr22_hmm_data.npz") as data:
        print(data.files)
        if "ref_genotypes" in data.files:
            key = "ref_genotypes"
        else:
            # fall back to the first array if naming differs
            key = data.files[0]

        ref_matrix = data["ref_hap0"]
        print(f"using array '{key}' from chr22_hmm_data.npz")

    # -----------------------------
    # Load sample metadata
    # -----------------------------
    print("loading population data")
    kg_samples = np.genfromtxt(
        "data/igsr_samples.tsv",
        delimiter="\t",
        names=True,
        dtype=None,
        encoding=None,
    )

    # -----------------------------
    # Load reference sample order
    # -----------------------------
    print("loading reference sample order")
    with open("data/ref_samples.txt") as f:
        samples = [line.strip() for line in f]

    num_samples = len(samples)

    # Infer orientation of the reference matrix using the known number of samples.
    # We want a view where axis 0 is SNPs and axis 1 is samples (to match test.py).
    # if ref_matrix.shape[0] == num_samples:
    #     # Shape is (samples, snps) -> transpose to (snps, samples)
    #     ref_snps_by_sample = ref_matrix.T
    # elif ref_matrix.shape[1] == num_samples:
    #     # Shape is already (snps, samples)
    #     ref_snps_by_sample = ref_matrix
    # else:
    #     raise ValueError(
    #         f"Could not infer orientation of reference matrix: "
    #         f"shape={ref_matrix.shape}, num_samples={num_samples}"
    #     )
    ref_snps_by_sample = ref_matrix
    print(
        f"reference matrix loaded with "
        f"{ref_snps_by_sample.shape} SNPs and {ref_snps_by_sample.shape} samples"
    )

    # -----------------------------
    # Build ancestry labels for CEU/YRI
    # -----------------------------
    sample_to_pop = dict(
        zip(kg_samples["Sample_name"], kg_samples["Population_code"])
    )

    ancestry_code_to_label = {
        "CEU": 0,
        "YRI": 1,
    }

    keep_indices = []
    ancestry_labels = []

    for i, sample in enumerate(samples):
        pop_code = sample_to_pop.get(sample)
        if pop_code in ancestry_code_to_label:
            keep_indices.append(i)
            ancestry_labels.append(ancestry_code_to_label[pop_code])

    if len(keep_indices) < 2:
        raise ValueError(
            "Need at least two CEU/YRI donors to run BetterStatesLaihmm test."
        )

    keep_indices = np.array(keep_indices, dtype=int)
    ancestry_labels = np.array(ancestry_labels, dtype=int)

    print(
        f"using {len(keep_indices)} donors with CEU/YRI labels "
        f"out of {num_samples} total samples"
    )

    # -----------------------------
    # Choose target and remove from panel
    # -----------------------------
    # Use an arbitrary donor (first in the filtered set) as the target.
    target_pos = 0
    target_global_index = keep_indices[target_pos]
    target_label = ancestry_labels[target_pos]
    target_id = samples[target_global_index]

    print(
        f"using sample {target_id} as target "
        f"(ancestry label {target_label})"
    )

    # Extract target haplotype (phased data; handled as a single haplotype sequence)
    target_haplotype = ref_snps_by_sample[:, target_global_index]

    # Build donor panel by removing the target from the reference panel and labels
    donor_cols = np.delete(keep_indices, target_pos)
    donor_ancestry_labels = np.delete(ancestry_labels, target_pos)

    donor_panel_snps_by_sample = ref_snps_by_sample[:, donor_cols]
    # BetterStatesLaihmm expects shape (num_donors, num_snps)
    reference_panel = donor_panel_snps_by_sample.T

    print(
        f"reference panel for HMM: {reference_panel.shape[0]} donors, "
        f"{reference_panel.shape[1]} SNPs"
    )

    # -----------------------------
    # Run BetterStatesLaihmm
    # -----------------------------
    print("running BetterStatesLaihmm on held-out target")
    model = BetterStatesLaihmm(reference_panel, donor_ancestry_labels)
    ancestry_sequence = np.array(model.predict(target_haplotype))

    # -----------------------------
    # Evaluate under homogeneous-ancestry assumption
    # -----------------------------
    print("\nRESULTS FOR HELD-OUT HOMOGENEOUS TARGET:")
    true_ancestry = np.full_like(ancestry_sequence, target_label)

    errors = np.sum(ancestry_sequence != true_ancestry)
    accuracy = (len(ancestry_sequence) - errors) / len(ancestry_sequence)

    print(f"accuracy: {accuracy}")
    print(f"number of errors: {errors}")


if __name__ == "__main__":
    main()

