from neuralbec.simulation import VariableCouplingBec, Bec
from neuralbec.simulation import TwoComponentBec
from neuralbec.simulation import TwoDimensionalBec
from neuralbec.simulation import VariableCouplingTwoComponentBec
from neuralbec.simulation import VariableCouplingTwoDimBec
from neuralbec.visualize import build_visuals_from_file
from neuralbec.visualize import build_visuals_from_file2
from neuralbec.visualize import build_2d_visuals_from_file
from neuralbec.visualize import render_prediction_plot
from neuralbec.visualize import render_prediction_plot2
from neuralbec.visualize import render_prediction_plot2d
from neuralbec.visualize import make_prediction_plot
from neuralbec.visualize import plot_predictions_overlay
from neuralbec.approximations import make_testset
from neuralbec.approximations import fit, fit2, fit2d
from neuralbec.colors import color_list

from config import configs, setup

import argparse
import warnings

import os
import random

colors = random.sample(color_list, 20)

# ignore warnings
warnings.filterwarnings("ignore")

# [ .... config .... ]
parser = argparse.ArgumentParser(
    description='neuralbec : Neural Network based simulation of BEC'
    )
# ..
# o Positional Arguments
#     >> python3 main.py config
parser.add_argument('config', type=str, help="Configuration Name")
# ..
# o Optional Arguments
#     >> python3 main.py config --simulate --coupling=10.03
parser.add_argument('--simulate', default=False, action='store_true',
    help='Run simulation')
parser.add_argument('--coupling', type=float, default=None, help='Coupling Strength')
parser.add_argument('--couplings', type=str, default=None,
    help='Coupling Strengths')
parser.add_argument('--approximate', default=False, action='store_true',
    help='Create approximation on simulated data')
parser.add_argument('--ssc', type=int, default=250,
    help='(Subsample Count) Number of samples for fitting model')
parser.add_argument('--search', default=False, action='store_true',
    help='Optional switch for Hyperparameter Search')
parser.add_argument('--predict', default=False, action='store_true',
    help='Run prediction')
parser.add_argument('--model', type=str, default=None, help='Saved model file')
parser.add_argument('--visualize', default=False, action='store_true',
    help='Visualize results')
parser.add_argument('--save-to-file', default=False, action='store_true',
    help='Save figures to file')
parser.add_argument('--overlay', default=False, action='store_true',
    help='Visualize results')
parser.add_argument('--two-component', default=False, action='store_true',
    help='Working on 2 component systems')
parser.add_argument('--two-dim', default=False, action='store_true',
    help='Working on 2 dimensional systems')
parser.add_argument('--simulates', default=False, action='store_true',
    help='Run many simulations')
# ...
# parse command-line arguments
args = parser.parse_args()


def main_single_component():
  # ..
  #   [[[[[ MODE ]]]]]
  # ...
  # ^_^_^ Simulate ^_^_^
  # ...
  #   One-off
  if args.simulate and args.coupling:
    Bec(config, args.coupling).save(config)
  # ...
  #   Variable Coupling Values
  elif args.simulate and not args.coupling:
    VariableCouplingBec(config).run().save(config)
  # ...
  # v_v_v Approximate v_v_v
  # ...
  #   Fit a model on simulated data
  elif args.approximate and not args.search:
    model = fit(config, args.ssc)
  # ...
  #   Hyperparameter Search
  elif args.approximate and args.search:
    testset = make_testset(config, args.ssc)
    for ssc in config.sub_sample_counts:
      fit(config, ssc, testset=testset)
  # ...
  #   Prediction
  elif args.predict and args.model:
    assert args.couplings or args.coupling

    couplings = [args.coupling] if args.coupling else None
    if couplings is None:
      couplings = [ float(g) for g in args.couplings.replace(' ', '').split(',') ]
    data = []
    for i, coupling in enumerate(couplings):
      sample = Bec(config, coupling).df
      model = config.model(config=config).load(
          os.path.join(config.path_to_results, config.name, args.model))
      data.extend(make_prediction_plot(model, sample, (colors[i], colors[i + 1])))
      if not args.overlay:
        render_prediction_plot(model, sample)

    if args.overlay:
      plot_predictions_overlay(data)

  # ...
  #   **** >> Visualize **** >>
  elif args.visualize:
    path = os.path.join(config.path_to_results, config.name)
    if args.couplings:
      coupling_vars = [ float(g)
          for g in args.couplings.replace(' ', '').split(',') ]
      filename = 'sim_prediction.csv'
      VariableCouplingBec(config).run(coupling_vars).save(config, filename)
      build_visuals_from_file(
          os.path.join(path, filename), args.save_to_file, args.overlay)
    elif args.coupling:
      Bec(config, args.coupling).save(config)
      filename = 'sim_g={}.csv'.format(args.coupling)
      build_visuals_from_file(os.path.join(path, filename),
          args.save_to_file, args.overlay)
    else:
      files = os.listdir(path)
      for f in files:
        build_visuals_from_file(
            os.path.join(path, f), args.save_to_file, args.overlay)


def main_two_component():
  if args.simulate and not args.couplings:
    TwoComponentBec(config).save(config)
  elif args.simulates and not args.couplings:
    VariableCouplingTwoComponentBec(config).run().save(config)
  elif args.approximate and not args.search:
    model = fit2(config, args.ssc)
  elif args.visualize:
    build_visuals_from_file2(
        os.path.join(config.path_to_results, config.name, '2', 'sim.csv')
        )
  elif args.predict and args.model:
    couplings = config.couplings
    if args.couplings is not None:
      couplings = [ float(g) for g in args.couplings.replace(' ', '').split(',') ]
      assert len(couplings) == 3
    # simulate
    path = os.path.join(config.path_to_results, config.name, '2')
    filename = 'sim_prediction.csv'
    sample = TwoComponentBec(config, couplings).save(config, filename)
    # load model from file
    model = config.model(config=config).load(
        os.path.join(path, args.model))
    render_prediction_plot2(model, sample)
    """
    data.extend(make_prediction_plot(model, sample, (colors[i], colors[i+1])))
    if not args.overlay:
      render_prediction_plot(model, sample)

    if args.overlay:
      plot_predictions_overlay(data)
    """


def main_two_dim():
  if args.simulate and args.coupling:
    TwoDimensionalBec(config, args.coupling).save(config)
  elif args.simulate and not args.coupling:
    VariableCouplingTwoDimBec(config).run().save(config)
  elif args.visualize:
    path = os.path.join(config.path_to_results, config.name)
    files = os.listdir(path)
    for f in files:
      build_2d_visuals_from_file(os.path.join(path, f), args.save_to_file)
  elif args.approximate and not args.search:
    model = fit2d(config, args.ssc)
  elif args.predict and args.model:
    coupling = config.coupling if not args.coupling else args.coupling
    # simulate
    path = os.path.join(config.path_to_results, config.name)
    filename = 'sim_prediction.csv'
    sample = TwoDimensionalBec(config, coupling).save(config, filename)
    # load model from file
    model = config.model(config=config).load(os.path.join(path, args.model))
    render_prediction_plot2d(model, sample)


config = configs[args.config]
setup(config)  # init directories
if not args.two_component and not args.two_dim:
  main_single_component()
elif args.two_component:
  main_two_component()
elif args.two_dim:
  main_two_dim()
