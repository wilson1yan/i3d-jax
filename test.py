import pickle
import argparse
import numpy as np
import tensorflow_hub as hub
import jax

from i3d_jax import InceptionI3d


def main():
    tf_model_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
    flax_model = "i3d_jax/weights/i3d-kinetics-400.ckpt"
    
    # TF model
    tf_model = hub.load(tf_model_url)
    input_tensor = tf_model.graph.get_tensor_by_name("input_frames:0")
    tf_model = tf_model.prune(input_tensor, "RGB/inception_i3d/Mean:0")

    # Flax model
    with open(flax_model, "rb") as f:
        flax_state = pickle.load(f)
        flax_model = InceptionI3d()

    # Compare outputs
    x = np.random.randn(4, 16, 224, 224, 3)
    tf_logits, tf_out = tf_model(x)
    
    flax_logits, flax_out = flax_model.apply(
        flax_state,
        x,
        is_training=False
    )

    import ipdb; ipdb.set_trace()
    
    print("Passed")
    
    

if __name__ == '__main__':
    main()