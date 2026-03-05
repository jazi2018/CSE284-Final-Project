# CSE284-Final-Project
A python implementation of the FLARE local ancestry inference model detailed by Browning et al. 2023

## Installation
installation instructions

### How do I install this tool?


### What does this tool (currently) do?
This implementation takes an unphased genotype and returns 

### What are the valid inputs this tool works with?


### What kind of output will this tool provide?


## Model Details
The model detailed in Browning et al. 2023 utilizes a hidden markov model (HMM) to "paint" the genome with ancestral sources.

To simplify implementation, we make the assumption that, for every ancestry $a \in A$, there exists a reference panel $j \in J$ containing only samples of that ancestry. As such, $|J| = |A|$. The original FLARE model in Browing et al. offers the ability to make inferences on datasets where $|J| < |A|$ and $|J| > |A|$, but since the timeframe for implementation of this project is tight, this is a safe assumption which significantly reduces complexity.

We also make the assumption that data provided to the model comes from a SNP array in a VCF file, is phased, and comes with a genetic map which can be used to calculate distance in Morgans between SNPs.

### Observations / Emissions
Our observations are the sequence of alleles in some target individual's phased haplotype. The emission probability $P(O_m \mid S_m)$ depends on parameters which are learned through the Baum-Welch algorithm. More details can be found in the Supplementary Methods section of Browning et al. 2023.

### States
The unobserved state at marker $m$ is $S_m = (i, h)$, where $i$ is is the ancestry of $m$ and the donor reference haploytpe $h$.

### Transitions
There are two "types" of transitions in the FLARE model.
1. Transitioning from $h$ to $h'$, where $i$ remains the same.
2. Transitioning from $i$ to $i'$, where $h$ is guaranteed to change regardless.

To account for this, an ancestry specific switch rate $\rho_i$ is learned.

The overall probability of transitioning from state $(i, h)$ to $(i', h')$ is given by $P(S_m=(i',h')\mid S_{m-1} = (i,h))$. It's derivation is also provided within the Supplementary Methods section of the original paper.

### Initials
With an $A$ length vector $\mu$, the initial probability $\pi (i, h)$ is given by $\mu_i q_{ih}$ where $q_{ih} = p_{ij} / n_j$

## Future To-do's, Anticipated Challenges, and Desired Points of Feedback
- For full list of specific to-do items, see `TODO.md`
- What metric for measuring accuracy of the outcome is best? Most intuitive?
- Which FLARE optimizations should be prioritized or de-emphasized given time constraints?
- What sort of data input would you, dear reader, most want a tool to take when doing LAI?
  - Especially for someone who is less familiar with LAI, what data source/input type would you think is best?

## Citations
1. Browning SR, Waples RK, Browning BL. Fast, accurate local ancestry inference with FLARE. Am J Hum Genet. 2023 Feb 2;110(2):326-335. doi: [10.1016/j.ajhg.2022.12.010](https://doi.org/10.1016/j.ajhg.2022.12.010). Epub 2023 Jan 6. PMID: 36610402; PMCID: PMC9943733.
