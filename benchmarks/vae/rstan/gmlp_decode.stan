// Stan port of gMLPDeepRV(num_blks=2) with the default sub-MLP configs from
// benchmarks/vae/deep_rv_example.py:
//
//   embed    = MLP([64,  64],  gelu)
//   proj_in  = MLP([128, 128], gelu)
//   proj_out = MLP([64,  64],  gelu)
//   head     = MLP([128, 1],   gelu)
//   SGU.num_heads = 1
//   gate_fn  = identity
//   attn     = None
//
// Important Flax wrinkle: SpatialGatingUnit's default `norm = nn.LayerNorm()`
// is a class-level mutable default. Flax detects the same module instance
// across both blocks and ties their LayerNorm params. The Stan port mirrors
// that — sgu_norm_scale / sgu_norm_bias are applied in BOTH blocks.

functions {
  // GELU(x) = x * Phi(x) — exact form used by jax.nn.gelu(approximate=False).
  vector gelu_v(vector x) {
    int N = num_elements(x);
    vector[N] y;
    for (i in 1:N) y[i] = x[i] * Phi(x[i]);
    return y;
  }
  matrix gelu_m(matrix X) {
    int R = rows(X); int C = cols(X);
    matrix[R, C] Y;
    for (i in 1:R) for (j in 1:C) Y[i, j] = X[i, j] * Phi(X[i, j]);
    return Y;
  }

  // Flax LayerNorm with use_bias=use_scale=True, epsilon=1e-6, biased variance.
  matrix layer_norm(matrix X, vector gamma, vector beta_) {
    int R = rows(X); int C = cols(X);
    matrix[R, C] Y;
    real eps = 1e-6;
    for (i in 1:R) {
      row_vector[C] xi = row(X, i);
      real m = sum(xi) / C;
      real v = 0;
      for (j in 1:C) v += (xi[j] - m) * (xi[j] - m);
      v /= C;  // population variance (Flax divides by N)
      real inv_std = inv_sqrt(v + eps);
      for (j in 1:C) Y[i, j] = (xi[j] - m) * inv_std * gamma[j] + beta_[j];
    }
    return Y;
  }

  // Single Dense layer (no activation).
  matrix dense(matrix X, matrix W, vector b) {
    int R = rows(X); int Cout = cols(W);
    matrix[R, Cout] Y = X * W;
    for (i in 1:R) for (j in 1:Cout) Y[i, j] += b[j];
    return Y;
  }

  // SpatialGatingUnit(num_heads=1, gate_fn=identity).
  // x: (L, D_proj_in). Split last dim in half → z1, z2 each (L, D_gate).
  // z2' = LN(z2); z2'' = SGU_W (L, L) @ z2' + bias_row(L) broadcast over D_gate.
  // out = z1 * z2'' elementwise.
  matrix spatial_gating_unit(matrix x,
                             matrix sgu_W, vector sgu_b,
                             vector sgu_norm_scale, vector sgu_norm_bias) {
    int L = rows(x);
    int D = cols(x) %/% 2;
    matrix[L, D] z1;
    matrix[L, D] z2;
    for (i in 1:L) for (j in 1:D) {
      z1[i, j] = x[i, j];
      z2[i, j] = x[i, D + j];
    }
    z2 = layer_norm(z2, sgu_norm_scale, sgu_norm_bias);
    z2 = sgu_W * z2;
    for (i in 1:L) for (j in 1:D) z2[i, j] += sgu_b[i];
    matrix[L, D] out;
    for (i in 1:L) for (j in 1:D) out[i, j] = z1[i, j] * z2[i, j];
    return out;
  }

  // One gMLPBlock: proj_in (2-layer MLP w/ GELU) → SGU → proj_out (2-layer MLP w/ GELU).
  matrix gmlp_block(matrix x,
                    matrix pi_W0, vector pi_b0, matrix pi_W1, vector pi_b1,
                    matrix sgu_W, vector sgu_b,
                    vector sgu_norm_scale, vector sgu_norm_bias,
                    matrix po_W0, vector po_b0, matrix po_W1, vector po_b1) {
    matrix[rows(x), cols(pi_W0)] h = gelu_m(dense(x, pi_W0, pi_b0));
    matrix[rows(x), cols(pi_W1)] y = dense(h, pi_W1, pi_b1);          // (L, D_proj_in)
    matrix[rows(y), cols(y) %/% 2] g
        = spatial_gating_unit(y, sgu_W, sgu_b, sgu_norm_scale, sgu_norm_bias);
    matrix[rows(g), cols(po_W0)] u = gelu_m(dense(g, po_W0, po_b0));
    return dense(u, po_W1, po_b1);                                     // (L, D_embed)
  }

  // Full decoder forward.
  vector decode(vector z, vector cond, matrix s_mat,
                // embed
                matrix embed_W0, vector embed_b0, matrix embed_W1, vector embed_b1,
                // gMLP outer LayerNorms (before blk_0, before blk_1, before head)
                vector ln0_scale, vector ln0_bias,
                vector ln1_scale, vector ln1_bias,
                vector ln2_scale, vector ln2_bias,
                // shared SGU LayerNorm (tied across blocks)
                vector sgu_norm_scale, vector sgu_norm_bias,
                // block 0
                matrix b0_pi_W0, vector b0_pi_b0, matrix b0_pi_W1, vector b0_pi_b1,
                matrix b0_sgu_W, vector b0_sgu_b,
                matrix b0_po_W0, vector b0_po_b0, matrix b0_po_W1, vector b0_po_b1,
                // block 1
                matrix b1_pi_W0, vector b1_pi_b0, matrix b1_pi_W1, vector b1_pi_b1,
                matrix b1_sgu_W, vector b1_sgu_b,
                matrix b1_po_W0, vector b1_po_b0, matrix b1_po_W1, vector b1_po_b1,
                // head
                matrix head_W0, vector head_b0, matrix head_W1, vector head_b1) {
    int L = num_elements(z);
    int C = num_elements(cond);
    int F_IN = 1 + 2 + C;
    matrix[L, F_IN] x;
    for (i in 1:L) {
      x[i, 1] = z[i];
      x[i, 2] = s_mat[i, 1];
      x[i, 3] = s_mat[i, 2];
      for (k in 1:C) x[i, 3 + k] = cond[k];
    }
    matrix[L, cols(embed_W0)] e = gelu_m(dense(x, embed_W0, embed_b0));
    e = dense(e, embed_W1, embed_b1);                          // (L, 64)
    // Block 0
    matrix[L, cols(e)] y0 = gmlp_block(layer_norm(e, ln0_scale, ln0_bias),
                                       b0_pi_W0, b0_pi_b0, b0_pi_W1, b0_pi_b1,
                                       b0_sgu_W, b0_sgu_b,
                                       sgu_norm_scale, sgu_norm_bias,
                                       b0_po_W0, b0_po_b0, b0_po_W1, b0_po_b1);
    e = e + y0;
    // Block 1 (tied SGU norm)
    matrix[L, cols(e)] y1 = gmlp_block(layer_norm(e, ln1_scale, ln1_bias),
                                       b1_pi_W0, b1_pi_b0, b1_pi_W1, b1_pi_b1,
                                       b1_sgu_W, b1_sgu_b,
                                       sgu_norm_scale, sgu_norm_bias,
                                       b1_po_W0, b1_po_b0, b1_po_W1, b1_po_b1);
    e = e + y1;
    // Head
    matrix[L, cols(e)] e_n = layer_norm(e, ln2_scale, ln2_bias);
    matrix[L, cols(head_W0)] h = gelu_m(dense(e_n, head_W0, head_b0));
    matrix[L, 1] out_mat = dense(h, head_W1, head_b1);
    vector[L] out;
    for (i in 1:L) out[i] = out_mat[i, 1];
    return out;
  }
}

data {
  int<lower=1> L;
  int<lower=1> cond_dim;
  matrix[L, 2] s_mat;

  // embed (4 -> 64 -> 64)
  matrix[1 + 2 + cond_dim, 64] embed_W0;  vector[64] embed_b0;
  matrix[64, 64]               embed_W1;  vector[64] embed_b1;

  // gMLP outer LayerNorms
  vector[64] ln0_scale; vector[64] ln0_bias;
  vector[64] ln1_scale; vector[64] ln1_bias;
  vector[64] ln2_scale; vector[64] ln2_bias;

  // shared SGU LayerNorm (tied)
  vector[64] sgu_norm_scale; vector[64] sgu_norm_bias;

  // block 0
  matrix[64, 128]  b0_pi_W0;  vector[128] b0_pi_b0;
  matrix[128, 128] b0_pi_W1;  vector[128] b0_pi_b1;
  matrix[L, L]     b0_sgu_W;  vector[L]   b0_sgu_b;
  matrix[64, 64]   b0_po_W0;  vector[64]  b0_po_b0;
  matrix[64, 64]   b0_po_W1;  vector[64]  b0_po_b1;

  // block 1
  matrix[64, 128]  b1_pi_W0;  vector[128] b1_pi_b0;
  matrix[128, 128] b1_pi_W1;  vector[128] b1_pi_b1;
  matrix[L, L]     b1_sgu_W;  vector[L]   b1_sgu_b;
  matrix[64, 64]   b1_po_W0;  vector[64]  b1_po_b0;
  matrix[64, 64]   b1_po_W1;  vector[64]  b1_po_b1;

  // head (64 -> 128 -> 1)
  matrix[64, 128] head_W0;  vector[128] head_b0;
  matrix[128, 1]  head_W1;  vector[1]   head_b1;

  // Inference inputs (same shape as mlp_decode.stan).
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
