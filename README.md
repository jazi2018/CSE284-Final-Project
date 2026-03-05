# CSE284-Final-Project
A python implementation of the FLARE local ancestry inference model detailed by Browning et al. 2023

## Installation
To install this tool, start by cloning the repository:

``` bash
git clone https://github.com/jazi2018/CSE284-Final-Project.git
```
Navigate to the repository directory and (optional but recommended) create a virtual environment
``` bash
python -m venv env
```
Activate your virtual environment and install the required libraries
``` bash
pip install -r requirements.txt
```
Our model can be imported from `model.py`
``` python
from model import Laihmm
```
We've included a test script (feel free to modify) called `test.py`. It
1. Loads data from the data folder
2. Isolates two samples from different populations. One from CEU, one from YRI.
3. Generates an emission matrix for the CEU sample and runs the model on it. When calculating accuracy, we make the assumption that the CEU sample has undergone no admixture, and thus their entire genome is attributed to the CEU population.
4. Performs random admixture on the CEU and YRI population. Our implementation is very inelegant, and we plan on using haptools for future testing.
5. Generates an emission matrix for the new admixed sample on which the model is run. Accuracy is calculated off the true admixed genotype labels.

`test.py` is relatively slow to run, primarily limited by the time it takes to read in the data, and the time it takes to run the viterbi algorithm. Regardless, feel free to tinker with various parameters, such as the transition probability in the model or the switch probability in the admixing portion.

## Our current model
Due to the complexity of the FLARE model, we initially implement a simpler model (closer to the model described by Li and Stephens). We intend to implement the FLARE model (described below), but aimed to meet a minimum viable product goal for the peer-review session.

### Observations / Emissions
Our sequence of observations is simply the allele sequence in the target. The emission probability is the allele frequency of the emitted allele.

### States
The unobserved state at each snp is the ancestry from which the SNP is derived. The model can accept any number of ancestries.

### Transitions
We assume a constant transition probability $t$, such that the probability to switch to *any* other state is $t$. As such, the transition matrix is an $A \times t$ matrix, where each entry is $\frac{t}{A}$ and the diagonal is $1 - t$.

### Initials
We assume an equal probability to start in any initial state.

## FLARE Model Details
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
