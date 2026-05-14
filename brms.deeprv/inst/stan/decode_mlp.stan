// Stan port of MLPDeepRV(dims=[L, L]) forward pass.
//
// This file is the canonical `decode` function for v0.1. Mirrors:
//   dl4bi/vae/deep_rv.py:MLPDeepRV          (concat -> Dense -> relu -> Dense)
//   dl4bi/vae/train_utils.py:cond_as_locs   (concat z with broadcast cond)
//   dl4bi/core/mlp.py:MLP                   (Dense layers, default act_fn = relu)
//
// Used in three places inside brms.deeprv:
//   - install-time forward-parity tests (rstan::expose_stan_functions)
//   - inclusion as a `stanvars(block = "functions")` snippet at fit time
//   - downstream packages can inline the same source for prediction
//
// Bumping the body of this function REQUIRES bumping arch_version in
// the catalog config - otherwise old .rds artifacts will be treated as
// compatible by load_deeprv() despite a numerically different forward.

functions {
  // Single-sample forward pass: returns mu of length L given latent z (L)
  // and conditional hyperparameters cond (cond_dim).
  vector decode(vector z, vector cond,
                matrix W1, vector b1,
                matrix W2, vector b2) {
    int L = rows(z);
    int C = rows(cond);
    vector[L + C] x;
    for (i in 1:L) x[i] = z[i];
    for (j in 1:C) x[L + j] = cond[j];
    // Layer 1: hidden, with relu.
    vector[cols(W1)] h = W1' * x + b1;
    for (i in 1:rows(h)) h[i] = fmax(h[i], 0);
    // Layer 2: output, linear.
    return W2' * h + b2;
  }
}
