from typing import Tuple, Callable
import jax
import jax.numpy as jnp
import flax.linen as nn


VALID_ENDPOINTS = (
    'Conv3d_1a_7x7',
    'MaxPool3d_2a_3x3',
    'Conv3d_2b_1x1',
    'Conv3d_2c_3x3',
    'MaxPool3d_3a_3x3',
    'Mixed_3b',
    'Mixed_3c',
    'MaxPool3d_4a_3x3',
    'Mixed_4b',
    'Mixed_4c',
    'Mixed_4d',
    'Mixed_4e',
    'Mixed_4f',
    'MaxPool3d_5a_2x2',
    'Mixed_5b',
    'Mixed_5c',
    'Logits',
    'Predictions',
)


# https://github.com/deepmind/kinetics-i3d/blob/0667e889a5904b4dc122e0ded4c332f49f8df42c/i3d.py#L32
class Unit3D(nn.Module):
    output_channels: int 
    kernel_shape: Tuple[int] = (1, 1, 1)
    stride: Tuple[int] = (1, 1, 1)
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_batch_norm: bool = True
    use_bias: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, is_training: bool):
        net = nn.Conv(
            self.output_channels,
            self.kernel_shape,
            strides=self.stride,
            padding="SAME",
            use_bias=self.use_bias
        )(inputs)
        if self.use_batch_norm:
            # Match Sonnet v1 BatchNorm: https://github.com/deepmind/sonnet/blob/v1/sonnet/python/modules/batch_norm.py
            net = nn.BatchNorm(
                momentum=0.999,
                epsilon=1e-3,
                use_scale=False,
                use_running_average=not is_training
            )(net)
        if self.activation_fn is not None:
            net = self.activation_fn(net)
        return net

        
class InceptionModule(nn.Module):
    out_channels: Tuple[int]

    @nn.compact
    def __call__(self, net: jnp.ndarray, is_training: bool):
        # Branch 0
        branch_0 = Unit3D(
            output_channels=self.out_channels[0],
            kernel_shape=(1, 1, 1),
            name="Branch_0/Conv3d_0a_1x1"
        )(net, is_training=is_training)
        
        # Branch 1
        branch_1 = Unit3D(
            output_channels=self.out_channels[1],
            kernel_shape=(1, 1, 1),
            name="Branch_1/Conv3d_0a_1x1"
        )(net, is_training=is_training)
        branch_1 = Unit3D(
            output_channels=self.out_channels[2],
            kernel_shape=(3, 3, 3),
            name="Branch_1/Conv3d_0b_3x3"
        )(branch_1, is_training=is_training)

        # Branch 2
        branch_2 = Unit3D(
            output_channels=self.out_channels[3],
            kernel_shape=(1, 1, 1),
            name="Branch_2/Conv3d_0a_1x1"
        )(net, is_training=is_training)
        branch_2 = Unit3D(
            output_channels=self.out_channels[4],
            kernel_shape=(3, 3, 3),
            name="Branch_2/Conv3d_0b_3x3"
        )(branch_2, is_training=is_training)

        # Branch 3
        branch_3 = nn.max_pool(
            net,
            window_shape=(3, 3, 3),
            strides=(1, 1, 1),
            padding="SAME"
        )
        branch_3 = Unit3D(
            output_channels=self.out_channels[5],
            kernel_shape=(1, 1, 1),
            name="Branch_3/Conv3d_0b_1x1"
        )(branch_3, is_training=is_training)

        return jnp.concatenate([
            branch_0, branch_1, branch_2, branch_3
        ], axis=-1)
        


# https://github.com/deepmind/kinetics-i3d/blob/0667e889a5904b4dc122e0ded4c332f49f8df42c/i3d.py#L74
class InceptionI3d(nn.Module):
    num_classes: int = 400
    spatial_squeeze: bool = True
    final_endpoint: str = "Logits"

    @nn.compact
    def __call__(
        self, inputs: jnp.ndarray, is_training: bool = False, dropout_keep_prob: float = 1.0
    ):
        # Inputs: B x T x 224 x 224 x 3 in range [-1, 1]
        if self.final_endpoint not in VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint {self.final_endpoint}") 
        
        net = inputs
        end_points = {}
        end_point = "Conv3d_1a_7x7"
        net = Unit3D(
            output_channels=64,
            kernel_shape=(7, 7, 7),
            stride=(2, 2, 2),
            name=end_point
        )(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "MaxPool3d_2a_3x3"
        net = nn.max_pool(
            net,
            window_shape=(1, 3, 3),
            strides=(1, 2, 2),
            padding="SAME"
        )
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "Conv3d_2b_1x1"
        net = Unit3D(
            output_channels=64,
            kernel_shape=(1, 1, 1),
            name=end_point
        )(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "Conv3d_2c_3x3"
        net = Unit3D(
            output_channels=192,
            kernel_shape=(3, 3, 3),
            name=end_point
        )(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "MaxPool3d_3a_3x3"
        net = nn.max_pool(
            net,
            window_shape=(1, 3, 3),
            strides=(1, 2, 2),
            padding="SAME"
        )
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "Mixed_3b"
        net = InceptionModule(
            [64, 96, 128, 16, 32, 32],
            name=end_point
        )(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "Mixed_3c"
        net = InceptionModule(
            [128, 128, 192, 32, 96, 64],
            name=end_point
        )(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "MaxPool3d_4a_3x3"
        net = nn.max_pool(
            net,
            window_shape=(3, 3, 3),
            strides=(2, 2, 2),
            padding="SAME"
        )
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "Mixed_4b"
        net = InceptionModule(
            [192, 96, 208, 16, 48, 64],
            name=end_point
        )(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points
        
        end_point = "Mixed_4c"
        net = InceptionModule(
            [160, 112, 224, 24, 64, 64],
            name=end_point
        )(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "Mixed_4d"
        net = InceptionModule(
            [128, 128, 256, 24, 64, 64],
            name=end_point
        )(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "Mixed_4e"
        net = InceptionModule(
            [112, 144, 288, 32, 64, 64],
            name=end_point
        )(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "Mixed_4f"
        net = InceptionModule(
            [256, 160, 320, 32, 128, 128],
            name=end_point
        )(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "MaxPool3d_5a_2x2"
        net = nn.max_pool(
            net,
            window_shape=(2, 2, 2),
            strides=(2, 2, 2),
            padding="SAME"
        )
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points
 
        end_point = "Mixed_5b"
        net = InceptionModule(
            [256, 160, 320, 32, 128, 128],
            name=end_point
        )(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "Mixed_5c"
        net = InceptionModule(
            [384, 192, 384, 48, 128, 128],
            name=end_point
        )(net, is_training=is_training)
        end_points[end_point] = net
        if self.final_endpoint == end_point: return net, end_points

        end_point = "Logits"
        net = nn.avg_pool(
            net,
            window_shape=(2, 7, 7),
            strides=(1, 1, 1),
            padding="VALID"
        )
        net = nn.Dropout(1. - dropout_keep_prob, deterministic=not is_training)(net)
        logits = Unit3D(
            output_channels=self.num_classes,
            kernel_shape=(1, 1, 1),
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name=f"{end_point}/Conv3d_0c_1x1"
        )(net, is_training=is_training)
        if self.spatial_squeeze:
            logits = jnp.squeeze(logits, [2, 3])
        averaged_logits = jnp.mean(logits, axis=1)        
        end_points[end_point] = averaged_logits
        if self.final_endpoint == end_point: return averaged_logits, end_points

        end_point = "Predictions"
        predictions = jax.nn.softmax(averaged_logits)
        end_points[end_point] = predictions
        return predictions, end_points
