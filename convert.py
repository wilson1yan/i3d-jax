import argparse
import pickle
import os
import os.path as osp
import numpy as np
import tensorflow_hub as hub


def main(args):
    tf_vars = hub.load(args.model_url).variables
    names = [v.name for v in tf_vars]
    names = [name[len("RGB/inception_i3d/"):] for name in names]

    # Convert to mapping TF name -> numpy array
    tf_vars = {name: tf_var.value().numpy() 
               for name, tf_var in zip(names, tf_vars)}
    flax_vars, flax_batch_stats = {}, {}

    def get_remove(name):
        assert tf_vars.get(name, None) is not None, f"{name} not found"
        weight = tf_vars[name]
        tf_vars[name] = None
        return weight

    def unit3d(prefix, use_bias=False, use_batch_norm=True):
        conv = {"kernel": get_remove(f"{prefix}/conv_3d/w:0")}
        if use_bias:
            conv["bias"] = get_remove(f"{prefix}/conv_3d/b:0")
        vars = {"Conv_0": conv}

        batch_stats = {} 
        if use_batch_norm:
            vars["BatchNorm_0"] = {"bias": np.squeeze(get_remove(f"{prefix}/batch_norm/beta:0"))}
            batch_stats["BatchNorm_0"] = {
                "mean": get_remove(f"{prefix}/batch_norm/moving_mean:0"),
                "var": get_remove(f"{prefix}/batch_norm/moving_variance:0")
            }
        
        return vars, batch_stats

    def inception_module(prefix):
        vars, batch_stats = {}, {}

        def update(sub_prefix):
            # Typo in original: https://github.com/deepmind/kinetics-i3d/blob/0667e889a5904b4dc122e0ded4c332f49f8df42c/i3d.py#L417
            if prefix == "Mixed_5b" and sub_prefix == "Branch_2/Conv3d_0b_3x3":
                v, b = unit3d(f"{prefix}/Branch_2/Conv3d_0a_3x3")
            else:
                v, b = unit3d(f"{prefix}/{sub_prefix}")
            vars[sub_prefix] = v
            batch_stats[sub_prefix] = b

        # Branch 0
        update("Branch_0/Conv3d_0a_1x1")

        # Branch 1
        update("Branch_1/Conv3d_0a_1x1")
        update("Branch_1/Conv3d_0b_3x3")

        # Branch 2
        update("Branch_2/Conv3d_0a_1x1")
        update("Branch_2/Conv3d_0b_3x3")

        # Branch 3
        update("Branch_3/Conv3d_0b_1x1")

        return vars, batch_stats


    def add(prefix, module, **kwargs):
        vars, batch_stats = module(prefix, **kwargs)
        flax_vars[prefix] = vars
        if len(batch_stats) > 0:
            flax_batch_stats[prefix] = batch_stats

    add("Conv3d_1a_7x7", unit3d)
    add("Conv3d_2b_1x1", unit3d)
    add("Conv3d_2c_3x3", unit3d)

    add("Mixed_3b", inception_module)
    add("Mixed_3c", inception_module)
    add("Mixed_4b", inception_module)
    add("Mixed_4c", inception_module)
    add("Mixed_4d", inception_module)
    add("Mixed_4e", inception_module)
    add("Mixed_4f", inception_module)
    add("Mixed_5b", inception_module)
    add("Mixed_5c", inception_module)

    add("Logits/Conv3d_0c_1x1", unit3d, 
        use_bias=True, use_batch_norm=False)

    assert all([v is None for v in tf_vars.values()])

    state = {"params": flax_vars, "batch_stats": flax_batch_stats}
    os.makedirs(osp.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(state, f) 
    print("Saved converted file to", args.output)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_url", type=str, 
        default="https://tfhub.dev/deepmind/i3d-kinetics-400/1", 
        help="TF Hub URL to checkpoint that should be converted to Flax"
    )
    parser.add_argument(
        "-o", "--output", type=str, 
        default="i3d_jax/weights/i3d-kinetics-400.ckpt", 
        help="Path to save converted checkpoint"
    )
    args = parser.parse_args()
    main(args)
