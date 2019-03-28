# Neural BEC

Machine Learning for simulating BEC systems


## Data Generation

```python
from neuralbec import data

data.generate_varg(
    fn=lambda g : data.particle_density_BEC1D(
      dim=512, radius=24, angular_momentum=1,
      time_step=1e-4, coupling=g,
      iterations=10000
      ),
    num_samples=10, filename='10samples.data'
    )
```

## Training

```bash
python3 main.py --train --model='ffn' --data='bec1d'
```

## Prediction

```bash
python3 main.py --predict --model='ffn' --data='bec1d'
```
