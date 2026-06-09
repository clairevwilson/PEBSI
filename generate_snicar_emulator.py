"""
SNICAR Emulator - Training
===========================
Usage:
    python train_snicar_emulator.py
    python train_snicar_emulator.py --data_dir snicar_data --epochs 100 --lr 1e-3
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from pathlib import Path
jax.config.update('jax_enable_x64', True)


# ============================================================
# MODEL
# ============================================================

class SNICAREmulator(eqx.Module):
    layers: list

    def __init__(self, in_dim, key):
        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(in_dim, 128, key=keys[0]), eqx.nn.LayerNorm((128,)),
            eqx.nn.Linear(128,    128, key=keys[1]), eqx.nn.LayerNorm((128,)),
            eqx.nn.Linear(128,     64, key=keys[2]), eqx.nn.LayerNorm((64,)),
            eqx.nn.Linear(64,       1, key=keys[3]),
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x)) if isinstance(layer, eqx.nn.Linear) else layer(x)
        return jax.nn.sigmoid(self.layers[-1](x)).squeeze()


# ============================================================
# TRAINING
# ============================================================

@eqx.filter_jit
def step(model, opt_state, X, y, optimizer):
    def loss_fn(model):
        pred = jax.vmap(model)(X)
        return jnp.mean((pred - y) ** 2)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train(data_dir, epochs, lr, batch_size, seed):
    # load data
    train = np.load(Path(data_dir) / 'snicar_train.npz')
    val   = np.load(Path(data_dir) / 'snicar_val.npz')
    X_train, y_train = jnp.array(train['X']), jnp.array(train['y'])
    X_val,   y_val   = jnp.array(val['X']),   jnp.array(val['y'])

    # normalize inputs
    mu, sigma = X_train.mean(0), X_train.std(0) + 1e-6
    X_train = (X_train - mu) / sigma
    X_val   = (X_val   - mu) / sigma

    # init model and optimizer
    model = SNICAREmulator(X_train.shape[1], jax.random.PRNGKey(seed))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    rng = np.random.default_rng(seed)
    n_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        # shuffle
        idx = rng.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

        for b in range(n_batches):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            model, opt_state, loss = step(model, opt_state, X_train[sl], y_train[sl], optimizer)

        if (epoch + 1) % 10 == 0:
            val_pred = jax.vmap(model)(X_val)
            val_rmse = jnp.sqrt(jnp.mean((val_pred - y_val) ** 2))
            print(f'Epoch {epoch+1:3d}  train_loss={loss:.6f}  val_rmse={val_rmse:.4f}')

    # save model and normalization stats
    eqx.tree_serialise_leaves('snicar_emulator.eqx', model)
    np.savez('snicar_norm.npz', mu=np.array(mu), sigma=np.array(sigma))
    print('Saved snicar_emulator.eqx and snicar_norm.npz')
    return model, mu, sigma


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str,   default='snicar_data')
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int,   default=1024)
    parser.add_argument('--seed',       type=int,   default=0)
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.lr, args.batch_size, args.seed)