# Neural BEC

Simulating the dynamics of Bose Einstein Condensates using Gaussian Processes.


## Simulation

Set configuration at `config.py`.

```python
# -------------------------
# ------- config.py ------- 


class ConfigHarmonic:
  dim = 512
  radius = 24
  angular_momentum = 1
  time_step = 1e-4
  coupling = 1
  potential_fn = None
  iterations = 10000
  coupling_vars = [1, 300]  # , 100]
  name = 'harmonic'

  def wave_function(r):  # constant
    return 1. / np.sqrt(24)

  def potential_fn(x, y):  # harmonic potential
    return 0.5 * (x ** 2 + y ** 2)
```

Run `main.py` with `--simulate` switch enabled.

```bash
python3 main.py --simulate
```

Data will be saved to `results/bec_harmonic.csv`
