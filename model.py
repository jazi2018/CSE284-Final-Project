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

        self.log_transitions = log_transitions
        ################################
    
    def predict(self) -> list[int]:
        """
        Predicts the most likely state sequence through the Viterbi algorithm.
        
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

        log_emissions = np.log(self.emissions)
        #initials - equal to 1 / |A| (each ancestry is equally likely)
        for idx in range(self.num_ancestries):
            #all operations are done in log space to prevent underflow (and make arithmetic easier)
            log_probs[idx][0] = np.log(1 / self.num_ancestries) + log_emissions[idx, 0]
        

        #Viterbi forward pass
        for t in tqdm(range(self.num_snps - 1)):
            #score[i, j] = log P(state i at SNP t) + log P(transitioning from state i to state j)
            scores = log_probs[:, t][:, None] + self.log_transitions

            #now we find the best previous state for each state in our scores table
            #for backtracking later
            backtrack[:, t + 1] = np.argmax(scores, axis=0)

            #and now find the best probability at each state
            log_probs[:, t + 1] = np.max(scores, axis=0) + log_emissions[:, t + 1]
        
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

        self.log_transitions = log_transitions

class BetterStatesLaihmm():
    def __init__(self, reference_panel: np.ndarray,
                ancestry_labels: list[int],
                recombination_prob: float = 0.01,
                admixture_prob: float = 0.001) -> None:
        """
        Initializes the model.

        Parameters
        -
        reference_panel : np.ndarray
            Shape: (Number of reference donors, number of SNPs)
            Values represent the haplotype at each SNP for each reference donor.
        
        ancestry_labels : list[int]
            A list of integers mapping each row of the reference panel to an ancestry.

        recombination_prob : float
            The probability of recombination between SNPs. In the model, this is the probability of switching
            to a different donor *within* the same ancestry. Defaults to 0.01.
        
        admixture_prob : float
            The probability of admixture at any given SNP. This represents the probability of transitioning to a
            different donor *between* ancestries. Defaults to 0.001.
        """
        #set basic attributes
        self.reference_panel = reference_panel
        self.ancestry_labels = ancestry_labels
        self.recombination_prob = recombination_prob
        self.admixture_prob = admixture_prob

        #set attributes which are derived from the input
        self.num_donors, self.num_snps = reference_panel.shape
        self.ancestries = np.unique(ancestry_labels) #we need to store unqiue ancestries since we use
        #only one emission matrix for all ancestries
        self.num_ancestries = len(self.ancestries)

        ### BUILD TRANSITION MATRIX ###
        #Li and Stephens / FLARE doesnt store an entire transition matrix,
        #because a large panel would contain a massive number of entries and require
        #many calculations to build - we can instead leverage the observation that
        #transition probabilities are only dependent on the ancestry of the donor (in this implementation)
        #and store ancestry specific transition probabilities instead
        self.log_recomb_probs = {}
        self.log_admixture_probs = {}
        self.log_self_transition_prob = np.log(1 - self.recombination_prob - self.admixture_prob)
        
        #get counts of number of donors in each ancestry
        donor_counts = {a: np.sum(self.ancestry_labels == a) for a in self.ancestries}

        for a in self.ancestries:
            num_same_ancestry = donor_counts[a] - 1
            num_different_ancestry = self.num_donors - donor_counts[a]

            #using 1e-100 (a very small epsilon) to handle impossible cases to prevent log(0) errors
            #and ternary operator to prevent division by zero errors
            self.log_recomb_probs[a] = np.log(self.recombination_prob / num_same_ancestry) if num_same_ancestry > 0 else np.log(1e-100)
            self.log_admixture_probs[a] = np.log(self.admixture_prob / num_different_ancestry) if num_different_ancestry > 0 else np.log(1e-100)
        ################################

        #old transition matrix code (for reference)
        # for i in range(self.num_donors): #iterate over each donor
        #     #log current donor's ancestry
        #     ancestry_i = self.ancestry_labels[i]
        #     #number of other donors with the same ancestry
        #     num_same_ancestry = donor_counts[ancestry_i] - 1
        #     #number of other donors with different ancestries
        #     num_different_ancestry = self.num_donors - donor_counts[ancestry_i]
            
        #     for j in range(self.num_donors): #iterate over each other donor
        #         ancestry_j = self.ancestry_labels[j]

        #         if i == j: #we're looking at the same donor
        #             #probability of staying on the same donor
        #             prob = 1.0 - (self.recombination_prob + self.admixture_prob)
        #         elif ancestry_i == ancestry_j: #we're looking at the same ancestry
        #             #recombination comes from the same ancestry so we use the recombination probability
        #             #ternary operator to handle case where there are no other donors with the same ancestry
        #             prob = self.recombination_prob / num_same_ancestry if num_same_ancestry > 0 else 0.0
        #         else: #we're looking at a different ancestry
        #             #admixture comes from a different ancestry so we use the admixture probability
        #             #ternary operator to handle case where there are no other donors with a different ancestry
        #                 #(i include the check just in case, but it shouldn't happen unless we only have one ancestry)
        #             prob = self.admixture_prob / num_different_ancestry if num_different_ancestry > 0 else 0.0
                
        #         #we keep it in log space to prevent underflow (and make arithmetic easier)
        #         log_transitions[i, j] = np.log(prob) if prob > 0 else -np.inf #log(0) is undefined
        
        # self.log_transitions = log_transitions
        ################################
    
    def predict(self, target_haplotype: np.ndarray, error_rate: float = 0.01) -> list[int]:
        """
        Predicts the most likely state sequence through an optimized Viterbi algorithm.
        
        Parameters
        -
        target_haplotype : np.ndarray
            The haplotype of the target individual to be predicted.
            Shape: (number of SNPs)
            Values represent the genotype at each SNP.
        
        error_rate : float
            The probability of error / mutation at any given SNP. This is necessary for emission calculations
            in the FLARE model.
            Defaults to 0.01.
    
        Returns
        -
        sequence : list[int]
            A list of predicted ancestries at each SNP, enumerated 0 through |A| - 1 in the
            order of the emission matrix provided in the initialization of the model.
            E.g. [0, 0, 0, 1, 1, 2, 2, 0, ...]

        Raises
        -
        ValueError
            If the target haplotype is not the same length as the reference panel.
        """
        #check that the target haplotype is the same length as the reference panel
        if target_haplotype.shape[0] != self.num_snps:
            raise ValueError("Target haplotype must be the same length as the reference panel.")
        
        #initialize Viterbi matrix
        log_probs = np.zeros((self.num_donors, self.num_snps), dtype=float)
        backtrack = np.zeros((self.num_donors, self.num_snps), dtype=int)
        
        ### BUILD EMISISON MATRIX ###
        #make a boolean matrix for our target haplotype matching each donor's haplotype
        #true when reference allele for target haplotype matches reference allele for donor haplotype
        is_match = (self.reference_panel == target_haplotype)

        #for every match to some donor haplotype, the probability of emitting the target allele
        #is 1 - our arbitrary error rate
        #for every mismatch, the probability of emitting the target allele is our error rate
        emissions = np.where(is_match, 1 - error_rate, error_rate)
        #then we want to work in log space
        log_emissions = np.log(emissions)
        ################################

        #calculate initials - we assume equal probability for each donor
        log_probs[:, 0] = np.log(1 / self.num_donors) + log_emissions[:, 0]

        #viterbi forward pass
        for t in tqdm(range(self.num_snps - 1)):
            prev_probs = log_probs[:, t]
            #we need to find the best donor for each ancestry
            best_in_ancestry = {}
            for a in self.ancestries:
                indices = (self.ancestry_labels == a)
                ancestry_probs = np.where(indices, prev_probs, -np.inf)

                best_in_ancestry[a] = np.argmax(ancestry_probs)
            
            #now need to calculate new probabilities from each donor
            for j in range(self.num_donors):
                ancestry_j = self.ancestry_labels[j]
                #there are three posible cases:
                    #we stay on the same donor
                    #we recombine from the same ancestry
                    #we admix from a different ancestry
                
                #self transition:
                score_self = prev_probs[j] + self.log_self_transition_prob
                
                #recombination:
                #saved an index so need to get probability from that donor
                score_recomb = prev_probs[best_in_ancestry[ancestry_j]] + self.log_recomb_probs[ancestry_j]

                #admixture:
                score_admixture = -np.inf
                best_admixture_donor = None
                #find the best donor in each ancestry
                for a in self.ancestries:
                    if a == ancestry_j:
                        continue
                    score = prev_probs[best_in_ancestry[a]] + self.log_admixture_probs[a]
                    if score > score_admixture:
                        score_admixture = score
                        best_admixture_donor = best_in_ancestry[a]
                
                #now find the best score from the three cases
                best = max(score_self, score_recomb, score_admixture)

                #and store probabilitiy in our matrix
                log_probs[j, t + 1] = best + log_emissions[j, t + 1]

                #and record backtracking information
                #check in this order just in case any scores are equal
                #logically, self transition would be most likely, followed by recombination, then admixture
                if best == score_self:
                    backtrack[j, t + 1] = j
                elif best == score_recomb:
                    backtrack[j, t + 1] = best_in_ancestry[ancestry_j]
                else:
                    backtrack[j, t + 1] = best_admixture_donor
        
        #backtracking to get state sequence
        state_sequence = np.zeros(self.num_snps, dtype=int)
        state_sequence[-1] = np.argmax(log_probs[:, -1])

        for t in range(self.num_snps - 1, 0, -1):
            state_sequence[t - 1] = backtrack[state_sequence[t], t]
        
        #map to an ancestry sequence and return
        ancestry_sequence = [self.ancestries[s] for s in state_sequence]
        return ancestry_sequence