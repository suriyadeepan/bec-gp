# Neural BEC

Machine Learning for simulating BEC systems


## How to generate data?

```python
data.generate_varg(
    fn=lambda g : data.particle_density_BEC1D(
      dim=512, radius=24, angular_momentum=1,
      time_step=1e-4, coupling=g,
      iterations=10000
      ),
    num_samples=10, filename='10samples.data'
    )
```
