import numpy as np
from tqdm import tqdm

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
    
    Initials:
    -
        We assume equal probability for any ancestry to be the initial sate. As such,
        πi for i ∈ A = 1 / |A|
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
                - All values are non-negative

        transition_prob : float
            A constant float which models the likelihood of transitioning between ancestries
            at any point. Defaults to 0.001.
        """
        self.emissions = emission_matrix
        self.transition_probability = transition_prob
        self.num_ancestries, self.num_snps = emission_matrix.shape
    
    def predict(self, target_genotype: np.ndarray):
        """
        Predicts the most likely state sequence through the Viterbi algorithm.

        Parameters
        -
        target_haplotype : np.array
            An array containing the phased haplotypes of some target for every SNP.

            Shape:
                (1, number of SNPs)
            
            Constraints:
                - n ∈ [0, 1] for n ∈ SNPs
                - All values are integers
        
        Returns
        -
        sequence : list[int]
            A list of predicted ancestries at each SNP, enumerated 0 through |A| - 1 in the
            order of the emission matrix provided in the initialization of the model.
            E.g. [0, 0, 0, 1, 1, 2, 2, 0, ...]
        """
        #initialize Viterbi matrix
        log_probs = np.zeros((self.num_ancestries, self.num_snps))

        #initials - equal to 1 / |A| (each ancestry is equally likely)
        for idx in range(self.num_ancestries):
            #all operations are done in log space to prevent underflow
            log_probs[idx][0] = np.log(1 / self.num_ancestries)
        
        log_emissions = np.log(self.emissions)
        log_transition = np.log(self.transition_probability)

        for t in range(self.num_snps - 1):
            snp = target_genotype[t + 1]
            #need to fix this broadcasting - it's for sure wrong (from my old code)
            #going to write it iteratively first and convert to broadcasting after cause thats
            #easier for me
            log_probs[:, t + 1] = np.max(log_probs[:, t][:, None] + log_transition, axis = 0) + log_emissions[:, snp]