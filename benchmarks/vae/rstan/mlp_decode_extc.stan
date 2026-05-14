// Same model as mlp_decode.stan, but `decode` is linked from
// mlp_decode_impl.hpp via Stan's --allow-undefined mechanism.

functions {
  vector decode(vector z, vector cond,
                matrix W1, vector b1,
                matrix W2, vector b2);
}

data {
  int<lower=1> L;
  int<lower=1> cond_dim;
  int<lower=1> hidden;
  matrix[L + cond_dim, hidden] W1;
  vector[hidden]              b1;
  matrix[hidden,         L]   W2;
  vector[L]                   b2;
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
      obs_idx[k] = i;
      y_obs[k] = y[i];
      k += 1;
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
  vector[L] mu = decode(z, cond, W1, b1, W2, b2);
}

model {
  z ~ std_normal();
  beta ~ std_normal();
  target += poisson_log_lpmf(y_obs | beta + mu[obs_idx]);
}
