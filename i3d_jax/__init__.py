from json import load
import os
import pickle
import inspect
import jax
import flax

from .i3d import InceptionI3d


def load_variables(replicate: bool = False):
        model_path = os.path.abspath(os.path.join(inspect.getfile(I3DWrapper.__init__), "..", "weights/i3d-kinetics-400.ckpt"))
        variables = pickle.load(open(model_path, "rb"))
        if replicate:
            variables = flax.jax_utils.replicate(variables)
        return variables
    

class I3DWrapper:
    def __init__(self, replicate: bool = True):
        self.i3d = InceptionI3d()
        self.variables = load_variables(replicate)
        self.replicate = replicate
    
    def __call__(self, video):
        # video: B x T x H x W x C in [-1, 1]
        fn = self.i3d.apply
        if self.replicate:
            fn = jax.pmap(fn)
        return fn(self.variables, video)
