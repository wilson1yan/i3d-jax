# I3D-Jax
Jax / Flax port of the original [Kinetics-400 I3D network](https://tfhub.dev/deepmind/i3d-kinetics-400/1) from TF

# Installation
`pip install i3d-jax`

# Usage
For convenience, we provide a wrapper to run inference on input videos
```python
import i3d_jax
import numpy as np

video = np.random.randn(1, 16, 224, 224, 3) # B x T x H x W x C in [-1, 1]
i3d = i3d_jax.I3DWrapper(replicate=False) # set to True to auto-use pmap

# out returns a tuple of:
# 1) logits
# 2) a dictionary mapping endpoint names to features at each endpoint
out = i3d(video)
```

You can separate get the model and variables through:
```python
import i3d_jax

# Load model
i3d_model = i3d_jax.InceptionI3d()

# Load variables (params + batch_stats)
variables = i3d_jax.load_variables(replicate=False)
```
