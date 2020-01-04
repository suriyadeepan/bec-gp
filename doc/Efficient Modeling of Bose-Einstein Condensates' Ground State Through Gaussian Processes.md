# Bayesian Modeling of Bose-Einstein Condensates' Ground State Through Gaussian Processes

- [ ] Introduction
  - Literature Survey
  - Proposal
- [ ] Gross Pitaevski Equation
  - Mean-field Theory Notation
  - One-dimensional BEC
    - One-component
    - Two-component
- [x] Method
  - Convolutional Neural Network
  - Gaussian Process
    - Why?
- [ ] Experiments
  - Solving One-Component and Two-Component BEC using GP
  - One-Component
    - Harmonic
      - Prediction vs Ground truth
      - 4 `g` values
      - Sub-sample Count Experiment
    - Double Well
    - Optical Lattice
  - Two-Component
    - Cosine
- [ ] Results
  - Table
    - Error vs Sub-sample count
      - Include previous work (50, 000 samples)
- [ ] Conclusion



# Method

In [1], Liang et al uses a deep Convolutional Neural Network to generate ground state wave function of Bose Einstein Condensates. The network learns a mapping between GPE parameters and ground state wave functions. The parameters of the network are learned using Stochastic Gradient Descent. The network is trained on wave functions generated through Imaginary Time Evolution by varying the parameters of GPE i.e. coupling strength. 

This work makes use of Gaussian Processes (GP) as an alternative to Liang et al's proposal, for modeling ground state wave functions of BEC. Our proposed method significantly reduces the number of data points (pre-generated wave functions) necessary for learning to generate the ground state wave functions. This is because GP by definition, models a probability distribution directly over functions while a neural network treats the problem as a vector to vector mapping. In addition to being natural and efficient modeling tool, GP inherently models uncertainty in available data.

We used an open-source implementation of Trotter-Suzuki to generate BEC ground state wave functions. We use them as training data for fitting GP. A comparison of number of data points used for training a GP and the number of data points used in Liang's work is presented in table [x]. 



## Experiments

We simulate ground-states of one-component BEC system using Trotter-Suzuki. Trotter-suzuki uses Imaginary Time Evolution to solve the ground state of GPE. We generate ground state wave functions by varying the coupling strength $$g$$ of the system. The generated wave functions along with the various coupling strength values are used as the training set for our model.

**Note** : Insert GPE equation here

The system consists of a one-dimensional lattice of 512 uniformly separated points, within the space $x \in (-10, 10)$. We set a *harmonic potential trap*. Every simulation takes $10^4$ iterations. 

From the simulated data points, we randomly sample (sub-sampling) 1000 points as training set, and another 1000 points as testset. Each data point is of the form $(x, g, \psi)$. We fit a Gaussian Process (GP) using the training set which learns a probability distribution over all functions that represent the data points in our training set. We compare the ground-truth wave functions from our simulations with the wave functions predicted by a trained GP for a given set of coupling strength $g$ values, figure x. We achieve a Mean-Squarred Error (MSE) in the  $10^{-7}$.

Apart from the accuracy which is represented by MSE, GP being a bayesian method can intrinsically model uncertainty in itself. This uncertainty stems from incompleteness in the training set exposed to GP. The training set used to fit our model, is a noisy representation of the real data distribution, as is the case for most data-driven systems. We present this uncertainty in the figure x, as a 95% confidence interval.

We have experimented with different sub-sample counts for training the GP and compared the performance of the trained models on the test set. In table x, we compare GPs trained on different sub-sample counts $\{250, 500, 750, 1000\}$ and presents the results of evaluating them on a common test set. As expected, the MSE corresponding to the sub-sample counts reduces with increase in number of samples. The error saturates at $1000$ samples in the range of $10^{-7}$. **Compared to previous state of the art results [x], our method achieves comparable results with ($\frac{1}{50}$th) a small fraction of samples used by Liang et al.** Notice the reduction in uncertainty measures as the sub-sample count increases. 

We have replicated our experiments by changing the potential from harmonic to Double well and Optical Lattice potentials and studied the results. The ground-truth wave functions from simulations and predictions from GP are presented in figure x, along with MSE values. 

**Note** : Chemical Potential goes here

Similar to [x], we investigate further by trying to fit our model on two-component BEC ground states. In 2-component BEC the ground state is determined by coupling strength values corresponding to each component, given by $g_{11}$ and $g_{22}$, the coupling strength between the two components $g_{12}$ and Rabi coupling coefficient $\Omega$. 

**Note** : Insert 2-component GPE equation here

We simulate wave functions of two-component BEC by setting. We set the Rabi coupling coefficient close to zero ($\Omega=-1$). 

