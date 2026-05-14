// Stan port of MLPDeepRV(dims=[L, L]) forward pass.
//
// Mirrors:
//   dl4bi/vae/deep_rv.py:MLPDeepRV   (concat -> Dense -> relu -> Dense)
//   dl4bi/vae/train_utils.py:cond_as_locs   (concat z with broadcast cond)
//   dl4bi/core/mlp.py:MLP   (Dense layers, default act_fn = relu)
//
// Weights are passed in as `data`, populated by check_match.R from the
// JSON artifact produced by export_mlp_decoder.py.

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

data {
  int<lower=1> L;
  int<lower=1> cond_dim;
  int<lower=1> hidden;     // = dims[0]
  matrix[L + cond_dim, hidden] W1;
  vector[hidden]              b1;
  matrix[hidden,         L]   W2;
  vector[L]                   b2;
}

parameters {
  // Match the canonical inference model in deep_rv_example.py:
  //   z ~ N(0, I_L),  ls ~ Uniform(1, 100) (use the trained prior range)
  vector[L] z;
  real<lower=1, upper=100> ls;
}

transformed parameters {
  vector[cond_dim] cond;
  cond[1] = ls;
  vector[L] mu = decode(z, cond, W1, b1, W2, b2);
}

model {
  z ~ std_normal();
  // ls has implicit Uniform(1,100) prior from its bounds.
}
