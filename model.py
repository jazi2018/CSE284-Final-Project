import numpy as np

class Laihmm:
    """
    A very simplified version of the FLARE model as detailed by Browning et al. 2023.

    States:
    -
        Ancestry i ∈ A, where A is the set of all ancestries from the reference panel.
    
    Observations / Emissions:
    -
        The observations are the sequence of alleles in the target sample. Emission
        probabilities are simply equal to the allele frequency at some given SNP in
        the current ancestry.

    Transition:
    -
        A constant probability for swapping ancestry at any point.
    """
    def __init__(self, emission_matrix: np.ndarray, transition_prob: float = 0.001):
        """
        Initializes the model.

        Parameters
        -
        emission_matrix : np.ndarray
            A matrix containing the emission probability for each SNP in each ancestry.
            We assume that each value in the emission matrix contains the allele frequency,
            in each population, for the observed allele at that SNP.

            Shape:
                (Number of ancestries, number of SNPs)

            Constraints:
                - The sum over each column should equal one
                - All values are non-negative

        transition_prob : float
            A constant float which models the likelihood of transitioning between ancestries
            at any point. Defaults to 0.001.
        """
        self.emissions = emission_matrix
        self.transition_probability = transition_prob
        self.num_ancestries, self.num_snps = emission_matrix.shape