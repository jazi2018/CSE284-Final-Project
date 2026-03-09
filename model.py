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
        
        Raises
        -
        ValueError
            If each donor in the reference panel does not have an ancestry label.
        """
        #set basic attributes
        #using asarray for 2 reasons:
            #1. numpy operations are much faster
            #2. we save on memory usage since the panels and labels should not be modified
        self.reference_panel = np.asarray(reference_panel)
        self.ancestry_labels = np.asarray(ancestry_labels)
        self.recombination_prob = recombination_prob
        self.admixture_prob = admixture_prob

        #set attributes which are derived from the input
        self.num_donors, self.num_snps = self.reference_panel.shape

        #simple safety check just in case
        if self.ancestry_labels.shape[0] != self.num_donors:
            raise ValueError("Each donor in the reference panel must have an ancestry label.")

        self.ancestries, self.donor_ancestry_index = np.unique(self.ancestry_labels, return_inverse=True)

        #pre-compute a donors_by_ancestry to avoid recomputing a boolean mask *every* SNP
        self.donors_by_ancestry = [
            np.flatnonzero(self.donor_ancestry_index == ancestry_idx)
            for ancestry_idx in range(len(self.ancestries))
        ]
        self.log_initial_probability = -np.log(self.num_donors)
        self.negative_infinity = -np.inf
        #we need to store unqiue ancestries since we use
        #only one emission matrix for all ancestries
        self.num_ancestries = len(self.ancestries)

        ### BUILD TRANSITION MATRIX ###
        #Li and Stephens / FLARE doesnt store an entire transition matrix,
        #because a large panel would contain a massive number of entries and require
        #many calculations to build - we can instead leverage the observation that
        #transition probabilities are only dependent on the ancestry of the donor (in this implementation)
        #and store ancestry specific transition probabilities instead
        self.log_self_transition_prob = np.log(1 - self.recombination_prob - self.admixture_prob)
        donor_counts = np.array(
            [donor_indices.size for donor_indices in self.donors_by_ancestry],
            dtype=int,
        )
        same_ancestry_targets = donor_counts - 1
        different_ancestry_targets = self.num_donors - donor_counts

        self.log_recomb_probs = np.full(self.num_ancestries, self.negative_infinity, dtype=float)
        self.log_admixture_probs = np.full(self.num_ancestries, self.negative_infinity, dtype=float)

        valid_recomb = same_ancestry_targets > 0
        valid_admixture = different_ancestry_targets > 0
        self.log_recomb_probs[valid_recomb] = np.log(
            self.recombination_prob / same_ancestry_targets[valid_recomb]
        )
        self.log_admixture_probs[valid_admixture] = np.log(
            self.admixture_prob / different_ancestry_targets[valid_admixture]
        )
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

        log_match = np.log1p(-error_rate)
        log_mismatch = np.log(error_rate)

        #nifty memory saving optimization - still need to store entire backtrack matrix
        #but can only use current and prev columns
        backtrack = np.zeros((self.num_donors, self.num_snps), dtype=np.int32)
        prev_probs = self.log_initial_probability + np.where(
            self.reference_panel[:, 0] == target_haplotype[0],
            log_match,
            log_mismatch,
        )

        best_scores_by_ancestry = np.full(self.num_ancestries, self.negative_infinity, dtype=float)
        second_scores_by_ancestry = np.full(self.num_ancestries, self.negative_infinity, dtype=float)
        best_donors_by_ancestry = np.full(self.num_ancestries, -1, dtype=np.int32)
        second_donors_by_ancestry = np.full(self.num_ancestries, -1, dtype=np.int32)

        score_self = np.empty(self.num_donors, dtype=float)
        score_recomb = np.full(self.num_donors, self.negative_infinity, dtype=float)
        score_admixture = np.full(self.num_donors, self.negative_infinity, dtype=float)
        recomb_donors = np.full(self.num_donors, -1, dtype=np.int32)
        admixture_donors = np.full(self.num_donors, -1, dtype=np.int32)
        donor_indices = np.arange(self.num_donors, dtype=np.int32)

        #viterbi forward pass
        for t in tqdm(range(1, self.num_snps), total=self.num_snps - 1):
            for ancestry_idx, ancestry_donors in enumerate(self.donors_by_ancestry):
                ancestry_scores = prev_probs[ancestry_donors]
                top_count = min(2, ancestry_donors.size)
                top_local_indices = np.argpartition(ancestry_scores, -top_count)[-top_count:]
                top_local_indices = top_local_indices[np.argsort(ancestry_scores[top_local_indices])[::-1]]

                best_local_idx = top_local_indices[0]
                best_donor = ancestry_donors[best_local_idx]
                best_donors_by_ancestry[ancestry_idx] = best_donor
                best_scores_by_ancestry[ancestry_idx] = ancestry_scores[best_local_idx]

                if ancestry_donors.size > 1:
                    second_local_idx = top_local_indices[1]
                    second_donors_by_ancestry[ancestry_idx] = ancestry_donors[second_local_idx]
                    second_scores_by_ancestry[ancestry_idx] = ancestry_scores[second_local_idx]
                else:
                    second_donors_by_ancestry[ancestry_idx] = -1
                    second_scores_by_ancestry[ancestry_idx] = self.negative_infinity

            ancestry_admixture_scores = best_scores_by_ancestry + self.log_admixture_probs
            if self.num_ancestries > 1:
                top_count = min(2, self.num_ancestries)
                top_ancestry_indices = np.argpartition(ancestry_admixture_scores, -top_count)[-top_count:]
                top_ancestry_indices = top_ancestry_indices[
                    np.argsort(ancestry_admixture_scores[top_ancestry_indices])[::-1]
                ]
                best_admixture_ancestry = top_ancestry_indices[0]
                second_admixture_ancestry = top_ancestry_indices[1] if top_count > 1 else -1
            else:
                best_admixture_ancestry = -1
                second_admixture_ancestry = -1

            score_self[:] = prev_probs + self.log_self_transition_prob
            score_recomb.fill(self.negative_infinity)
            score_admixture.fill(self.negative_infinity)
            recomb_donors.fill(-1)
            admixture_donors.fill(-1)

            for ancestry_idx, ancestry_donors in enumerate(self.donors_by_ancestry):
                best_recomb_score = best_scores_by_ancestry[ancestry_idx] + self.log_recomb_probs[ancestry_idx]
                second_recomb_score = second_scores_by_ancestry[ancestry_idx] + self.log_recomb_probs[ancestry_idx]

                score_recomb[ancestry_donors] = best_recomb_score
                recomb_donors[ancestry_donors] = best_donors_by_ancestry[ancestry_idx]

                use_second_best = ancestry_donors == best_donors_by_ancestry[ancestry_idx]
                if np.any(use_second_best):
                    score_recomb[ancestry_donors[use_second_best]] = second_recomb_score
                    recomb_donors[ancestry_donors[use_second_best]] = second_donors_by_ancestry[ancestry_idx]

                admixture_ancestry = best_admixture_ancestry
                if ancestry_idx == best_admixture_ancestry:
                    admixture_ancestry = second_admixture_ancestry

                if admixture_ancestry != -1:
                    score_admixture[ancestry_donors] = ancestry_admixture_scores[admixture_ancestry]
                    admixture_donors[ancestry_donors] = best_donors_by_ancestry[admixture_ancestry]

            best_scores = score_self.copy()
            best_previous_donors = donor_indices.copy()

            use_recomb = score_recomb > best_scores
            best_scores[use_recomb] = score_recomb[use_recomb]
            best_previous_donors[use_recomb] = recomb_donors[use_recomb]

            use_admixture = score_admixture > best_scores
            best_scores[use_admixture] = score_admixture[use_admixture]
            best_previous_donors[use_admixture] = admixture_donors[use_admixture]

            backtrack[:, t] = best_previous_donors
            prev_probs = best_scores + np.where(
                self.reference_panel[:, t] == target_haplotype[t],
                log_match,
                log_mismatch,
            )

        #backtracking to get state sequence
        state_sequence = np.zeros(self.num_snps, dtype=np.int32)
        state_sequence[-1] = np.argmax(prev_probs)

        for t in range(self.num_snps - 1, 0, -1):
            state_sequence[t - 1] = backtrack[state_sequence[t], t]
        
        #map to an ancestry sequence and return
        ancestry_sequence = [self.ancestry_labels[s] for s in state_sequence]
        return ancestry_sequence
