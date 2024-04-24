import jax.numpy as jnp
import jax 
from jax import jit
import jax.lax as lax

def init_Conv2D(key, in_channels, out_channels, kernel_shape):
    key1, key2 = jax.random.split(key)
    kernel = jax.random.normal(key1, (out_channels, in_channels) + kernel_shape) * 0.1
    bias = jax.random.normal(key2, (1, out_channels, 1, 1)) * 0.1
    return dict(weights=kernel, bias=bias)   

def init_fc(key, input_dim, output_dim):
    key1, key2 = jax.random.split(key)
    weights = jax.random.normal(key1, (input_dim, output_dim)) * 0.1
    bias = jax.random.normal(key2, (output_dim,)) * 0.1
    return dict(weights=weights, bias=bias)

def forward_Conv2D(params, x, strides=(1, 1), padding='SAME'):
    return lax.conv(x, params['kernel'], window_strides=strides, padding=padding) + params['bias']

def forward_fc(params, x):
    return jnp.dot(x, params['weights']) + params['bias']

def init_GOL_convnet(key):
    keys = jax.random.split(key, 3)
    params = []
    params.append(init_Conv2D(keys[0], 1, 2, (3, 3)))
    params.append(init_Conv2D(keys[1], 2, 2, (1, 1)))
    params.append(init_Conv2D(keys[2], 2, 1, (1, 1)))
    return params

def forward_GOL_convnet(params, x):
    x = forward_Conv2D(params[0], x)
    x = jax.nn.relu(x)
    x = forward_Conv2D(params[1], x)
    x = jax.nn.relu(x)
    x = forward_Conv2D(params[2], x)
    x = jax.nn.sigmoid(x)
    return x

