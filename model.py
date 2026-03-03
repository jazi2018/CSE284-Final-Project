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

    def __init__(self, emission_matrix: np.ndarray, transition_prob: float = 0.001) -> None:
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

        ### BUILD TRANSITION MATRIX ###
        #probability to transition to self
        self_transition = 1 - self.transition_probability
        #probability to transition to any other
        other_transition = self.transition_probability / (self.num_ancestries - 1)
        log_transitions = np.full((self.num_ancestries, self.num_ancestries), np.log(other_transition))
        #fill out our transition matrix
        np.fill_diagonal(log_transitions, np.log(self_transition))

        self.log_transitons = log_transitions
        ################################
    
    def predict(self, target_genotype: np.ndarray) -> list[int]:
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
        log_probs = np.zeros((self.num_ancestries, self.num_snps), dtype=float)
        backtrack = np.zeros((self.num_ancestries, self.num_snps), dtype=int)

        #initials - equal to 1 / |A| (each ancestry is equally likely)
        for idx in range(self.num_ancestries):
            #all operations are done in log space to prevent underflow (and make arithmetic easier)
            log_probs[idx][0] = np.log(1 / self.num_ancestries)
        
        log_emissions = np.log(self.emissions)

        #Viterbi forward pass
        for t in tqdm(range(self.num_snps - 1)):
            snp = target_genotype[t + 1]
            
            #score[i, j] = log P(state i at SNP t) + log P(transitioning from state i to state j)
            scores = log_probs[:, t][:, None] + self.log_transitions

            #now we find the best previous state for each state in our scores table
            #for backtracking later
            backtrack[:, t + 1] = np.argmax(scores, axis=0)

            #and now find the best probability at each state
            log_probs[:, t + 1] = np.max(scores, axis=0) + log_emissions[:, snp]
        
        #backtracking to get sequence
        state_sequence = np.zeros(self.num_snps, dtype=int)
        state_sequence[-1] = np.argmax(log_probs[:, -1]) #find best final state

        for t in range(self.num_snps - 1, 0, -1): #iterate backwards
            #previous state informs backtrack row, t informs column
            #backtrack was built to encode the best path to each node
            state_sequence[t - 1] = backtrack[state_sequence[t], t]
        
        return state_sequence.tolist()
    
    def update_transition(self, transition_prob: float) -> None:
        """
        Updates transition probability and re-builds transition matrix.

        Parameters:
        -
        transition_prob : float
            The new transition probability for the model. There is no default
            value provided.
        """
        self.transition_probability = transition_prob

        #probability to transition to self
        self_transition = 1 - self.transition_probability
        #probability to transition to any other
        other_transition = self.transition_probability / (self.num_ancestries - 1)
        log_transitions = np.full((self.num_ancestries, self.num_ancestries), np.log(other_transition))
        #fill out our transition matrix
        np.fill_diagonal(log_transitions, np.log(self_transition))

        self.log_transitons = log_transitions