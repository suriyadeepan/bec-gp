# Bayesian Modeling of Bose-Einstein Condensates' Ground State Through Gaussian Processes

- [ ] Introduction
  - Why?
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