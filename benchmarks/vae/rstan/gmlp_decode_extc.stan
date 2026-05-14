// Same gMLPDeepRV(num_blks=2) model as gmlp_decode.stan, but `decode` is
// implemented externally in gmlp_decode_impl.hpp via Stan's
// `--allow-undefined` mechanism. Everything else (data block, parameters,
// model, transformed parameters) is identical.

functions {
  // Forward declaration only — no body. rstan links the implementation
  // from gmlp_decode_impl.hpp passed via `includes`.
  vector decode(vector z, vector cond, matrix s_mat,
                matrix embed_W0, vector embed_b0,
                matrix embed_W1, vector embed_b1,
                vector ln0_scale, vector ln0_bias,
                vector ln1_scale, vector ln1_bias,
                vector ln2_scale, vector ln2_bias,
                vector sgu_norm_scale, vector sgu_norm_bias,
                matrix b0_pi_W0, vector b0_pi_b0,
                matrix b0_pi_W1, vector b0_pi_b1,
                matrix b0_sgu_W, vector b0_sgu_b,
                matrix b0_po_W0, vector b0_po_b0,
                matrix b0_po_W1, vector b0_po_b1,
                matrix b1_pi_W0, vector b1_pi_b0,
                matrix b1_pi_W1, vector b1_pi_b1,
                matrix b1_sgu_W, vector b1_sgu_b,
                matrix b1_po_W0, vector b1_po_b0,
                matrix b1_po_W1, vector b1_po_b1,
                matrix head_W0, vector head_b0,
                matrix head_W1, vector head_b1);
}

data {
  int<lower=1> L;
  int<lower=1> cond_dim;
  matrix[L, 2] s_mat;

  matrix[1 + 2 + cond_dim, 64] embed_W0;  vector[64] embed_b0;
  matrix[64, 64]               embed_W1;  vector[64] embed_b1;

  vector[64] ln0_scale; vector[64] ln0_bias;
  vector[64] ln1_scale; vector[64] ln1_bias;
  vector[64] ln2_scale; vector[64] ln2_bias;

  vector[64] sgu_norm_scale; vector[64] sgu_norm_bias;

  matrix[64, 128]  b0_pi_W0;  vector[128] b0_pi_b0;
  matrix[128, 128] b0_pi_W1;  vector[128] b0_pi_b1;
  matrix[L, L]     b0_sgu_W;  vector[L]   b0_sgu_b;
  matrix[64, 64]   b0_po_W0;  vector[64]  b0_po_b0;
  matrix[64, 64]   b0_po_W1;  vector[64]  b0_po_b1;

  matrix[64, 128]  b1_pi_W0;  vector[128] b1_pi_b0;
  matrix[128, 128] b1_pi_W1;  vector[128] b1_pi_b1;
  matrix[L, L]     b1_sgu_W;  vector[L]   b1_sgu_b;
  matrix[64, 64]   b1_po_W0;  vector[64]  b1_po_b0;
  matrix[64, 64]   b1_po_W1;  vector[64]  b1_po_b1;

  matrix[64, 128] head_W0;  vector[128] head_b0;
  matrix[128, 1]  head_W1;  vector[1]   head_b1;

  array[L] int<lower=0> y;
  array[L] int<lower=0, upper=1> obs_mask;
}

transformed data {
  int n_obs = 0;
  for (i in 1:L) n_obs += obs_mask[i];
  array[n_obs] int obs_idx;
  array[n_obs] int y_obs;
  {
    int k = 1;
    for (i in 1:L) if (obs_mask[i] == 1) {
      obs_idx[k] = i; y_obs[k] = y[i]; k += 1;
    }
  }
}

parameters {
  vector[L] z;
  real<lower=1, upper=100> ls;
  real beta;
}

transformed parameters {
  vector[cond_dim] cond;
  cond[1] = ls;
  vector[L] mu = decode(z, cond, s_mat,
                        embed_W0, embed_b0, embed_W1, embed_b1,
                        ln0_scale, ln0_bias, ln1_scale, ln1_bias, ln2_scale, ln2_bias,
                        sgu_norm_scale, sgu_norm_bias,
                        b0_pi_W0, b0_pi_b0, b0_pi_W1, b0_pi_b1,
                        b0_sgu_W, b0_sgu_b,
                        b0_po_W0, b0_po_b0, b0_po_W1, b0_po_b1,
                        b1_pi_W0, b1_pi_b0, b1_pi_W1, b1_pi_b1,
                        b1_sgu_W, b1_sgu_b,
                        b1_po_W0, b1_po_b0, b1_po_W1, b1_po_b1,
                        head_W0, head_b0, head_W1, head_b1);
}

model {
  z ~ std_normal();
  beta ~ std_normal();
  target += poisson_log_lpmf(y_obs | beta + mu[obs_idx]);
}
