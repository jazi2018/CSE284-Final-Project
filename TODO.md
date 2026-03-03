### Jared
1. ~~Finish HMM implementation~~
2. ~~Implement viterbi algorithm to find best state sequence in HMM~~
6. Build small testing dataset
7. Optimize Viterbi - transition matrix can probably be removed becuase its stay vs any other (?)
    - This might not be necessary, since we plan on reworking the model anyways
3. Update transition probability to account for genetic distance (see $\rho$ in FLARE paper)
5. Add EM / Baum-Welch for parameter optimization (?)
4. Add haplotype donors / replace population frequency for emissions
    - Basically, update state space to account for ancestry AND donor haplotypes
    - This complicates the state space quite a bit which necessitates the next step
5. Implement basic FLARE computational tricks
    - 3 paths
    - Probabilities explained in supplementary section