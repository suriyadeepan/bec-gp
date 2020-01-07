from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
import gpytorch
import torch

from gpbec import gputils
from tqdm import tqdm


def gp(inputs, outputs):
  # setup likelihood
  likelihood = GaussianLikelihood(num_tasks=gputils.num_tasks(outputs))
  # setup GP model
  model = GaussianProcess(inputs, outputs, likelihood)
  # "Loss" for GPs - the marginal log likelihood
  mll = ExactMarginalLogLikelihood(likelihood, model)
  return model, mll


class GaussianProcess(gpytorch.models.ExactGP):

  def __init__(self, inputs, outputs, likelihood):
    super(GaussianProcess, self).__init__(
        inputs, outputs, likelihood)
    self.inputs = inputs
    self.outputs = outputs
    self.likelihood = likelihood
    # choose kernels
    self.mean_module, self.covar_module = gputils.choose_kernel(outputs)
    # setup optimizer
    #  Use the adam optimizer
    self.optimizer = torch.optim.Adam(
        [{'params': self.parameters()}, ], lr=0.1)

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return MultivariateNormal(mean_x, covar_x)

  def update(self, mll, inputs=None, outputs=None, iterations=50):

    if inputs is not None and outputs is not None:
      self.set_train_data(inputs=inputs, targets=outputs, strict=False)
      self.inputs = inputs
      self.outputs = outputs

    inputs = self.inputs
    outputs = self.outputs
    # Find optimal model hyperparameters
    self.train()
    self.likelihood.train()
    # handle to progress bar
    pbar = tqdm(range(iterations))
    for i in pbar:
      # Zero gradients from previous iteration
      self.optimizer.zero_grad()
      # Output from model
      prediction = self(inputs)
      # Calc loss and backprop gradients
      loss = -mll(prediction, outputs)
      loss.backward()
      msg = 'Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
          i + 1, iterations, loss.item(),
          self.covar_module.base_kernel.lengthscale.item(),
          self.likelihood.noise.item()
      )
      pbar.set_description(msg)
      self.optimizer.step()

  def predict(self, inputs):
    # Get into evaluation (predictive posterior) mode
    self.eval()
    self.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
      return self.likelihood(self(inputs))

  def sigma(self, inputs):
    prediction = self.predict(inputs)
    ub, lb = prediction.confidence_region()
    return torch.abs(ub - lb)


class LearningMachine:

  def __init__(self, params, inputs=['g', 'x'], outputs=['psi']):
    self.inputs = inputs  # { g : 1. } < sample input
    self.outputs = outputs
    self.params = params
    # generate test inputs, outputs from params
    ti, to = gputils.generate_test_io(params)
    # create and manage a GP
    # self.model, self.mll = gp(ti, to)
    self.model, self.mll = None, None

  def __call__(self, examples, iterations=100):
    train_x, train_y = gputils.torch_em(examples[:, :2], examples[:, -1])
    if self.model is None:
      self.model, self.mll = gp(train_x, train_y)
    # update GP with new examples
    self.model.update(self.mll, train_x, train_y, iterations=iterations)
    # generate uniform inputs
    uinputs = gputils.generate_uniform_inputs(self.params, self.inputs)
    # get uncertainty estimate
    return self.model.sigma(uinputs)
