from gpbec import utils
from gpbec.cache import cache, in_cache
from gpbec.utils import sim1c


class Simulator:

  def __init__(self, params, inputs):
    """Simulator

    Parameters
    ----------
    params : (dict) Dictionary of simulation parameters
    inputs : (dict) Experimental (variable) inputs to simulation

    """
    # attach params to simulator
    for name in params:
      setattr(self, name, params.get(name))
    # find an appropriate simulation function
    self.sim_fn = utils.resolve_sim_fn(params)
    self.params = params
    self.inputs = inputs

  def __call__(self, proposals):
    results = {}
    for proposal in proposals:
      print(proposal)
      # make proposal (dict) hashable
      key = utils.freeze(proposal)
      # resolve proposal to params
      # self.sim_fn
      params = utils.resolve_proposal(proposal, self.params)
      # check if experiment results exist in cache
      if in_cache(params):
        results[key] = cache(params)
      else:
        sim_out = self.sim_fn(params)  # run simulation
        # post-process results
        results[key] = utils.postsim_proc(sim_out, proposal)
        # add results to cache
        cache(params, content=results[key])
    return results

  def generate_proposals(self):
    # given variables inputs and params
    #  generate proposals
    self.proposals = utils.generate_proposals(self.inputs)
    return self.proposals

  def simulate(self):
    # generate proposals
    proposals = self.generate_proposals()
    return self(proposals)
