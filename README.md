# CSE284-Final-Project
A python implementation of the FLARE local ancestry inference model detailed by Browning et al. 2023

## Installation
To install this tool, start by cloning the repository:

``` bash
git clone https://github.com/jazi2018/CSE284-Final-Project.git
```
From here, the process diverges depending on whether the project was cloned locally or within a Datahub instance.
### Datahub installation
All installation requirements should be satisfied provided the Datahub instance is running on the CSE284 docker image. Otherwise, the local installation steps below should be followed.
### Local installation
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
from model import BetterStatesLaihmm, Laihmm
```
We've included a test script (feel free to modify) called `test_unified.py`. It
1. Loads data from the data folder
2. Processes data into a useable format for both `Laihmm` and `BetterStatesLaihmm`
3. Runs some simple tests to verify functionality and compare quality of predictions between the two models

There also exists a notebook version of it (`test_unified.ipynb`), which contains visuals comparing the two models, and discussion of some results.

*Important note*: If you have cloned the repository locally, we have *not* included `ipykernel` or `matplotlib` as requirements in `requirements.txt`. This is intentional, as neither are necessary for the model to run. If you wish to use `test_unified.ipynb`, and didn't clone this repository into an environment which enables the use of python notebooks, these libraries must first be installed.
```bash
pip install ipykernel matplotlib
```

## Laihmm
Due to the complexity of the FLARE model, we initially implement a simpler model (closer to the model described by Li and Stephens).

### Observations / Emissions
Our sequence of observations is simply the allele sequence in the target. The emission probability is the allele frequency of the emitted allele.

### States
The unobserved state at each SNP is the ancestry from which the SNP is derived. The model can accept any number of ancestries.

### Transitions
We assume a constant transition probability $t$, such that the probability to switch to *any* other state is $t$. As such, the transition matrix is an $A \times t$ matrix, where each entry is $\frac{t}{|A|}$ and the diagonal is $1 - t$.

### Initials
We assume an equal probability to start in any initial state.

## BetterStatesLaihmm
An improvement upon the previous model, which utilizes a wider state space to account for multiple donors across multiple ancestries.

### Observations / Initials
Our observations and initial probability distribution are identical to the previous model.

### States
The unobserved state at each SNP is the haplotype which is donating the SNP. Each haplotype is derived from a specific ancestry, and this model can support any number of ancestries.

### Transitions
We assume two scalar transition probabilities, $r$ and $t$. $r$ represents the recombination probability - the probability of transitioning from one haplotype donor to another, within the same ancestry. $t$ represents the admixture probability, where instead we transition to a haplotype donor in a different ancestry. The values can be the same, though in our model's defaults we assume $t$ to be an order of magnitude lower than $r$.

### Emissions
The emission probability at any given SNP is defined by a scalar error rate $\epsilon$. If the SNP in the target haplotype matches the SNP in the donor haplotype, the emission probability is $1 - \epsilon$. Otherwise, the emission probability is $\epsilon$. This is to account for potential errors in sequencing, either in the target or an individual in the reference panel.

## Citations
1. Browning SR, Waples RK, Browning BL. Fast, accurate local ancestry inference with FLARE. Am J Hum Genet. 2023 Feb 2;110(2):326-335. doi: [10.1016/j.ajhg.2022.12.010](https://doi.org/10.1016/j.ajhg.2022.12.010). Epub 2023 Jan 6. PMID: 36610402; PMCID: PMC9943733.
