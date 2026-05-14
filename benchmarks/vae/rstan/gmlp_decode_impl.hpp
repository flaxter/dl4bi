// External C++ implementation of the gMLPDeepRV(num_blks=2) `decode` function,
// linked into the Stan model via `--allow-undefined` + rstan's `includes` arg.
//
// The all-Stan port in gmlp_decode.stan uses scalar `for (i in 1:R, j in 1:C)`
// loops for gelu_m, layer_norm, and SGU bias broadcast — each (i, j) iteration
// creates a separate `stan::math::var` on the autodiff tape. For L = 256 across
// 2 gMLP blocks, that's ~500k+ tape entries per HMC leapfrog from those ops
// alone. The vectorized `stan::math` overloads (e.g. tanh on a whole matrix)
// collapse each op into one tape entry that does the entire matrix in a single
// callback. That's where the per-iter speedup comes from.

#ifndef GMLP_DECODE_IMPL_HPP
#define GMLP_DECODE_IMPL_HPP

#include <stan/math.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <ostream>

namespace gmlp_decode_impl_detail {

// Cached constant: sqrt(2 / pi).
inline double sqrt_2_over_pi() {
  static const double v = std::sqrt(2.0 / M_PI);
  return v;
}

// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Matches jax.nn.gelu(approximate=True) — same form used in the all-Stan port.
template <typename T>
inline Eigen::Matrix<T, -1, -1> gelu_m(const Eigen::Matrix<T, -1, -1>& X) {
  using stan::math::tanh;
  using stan::math::elt_multiply;
  using stan::math::add;
  using stan::math::multiply;
  const double c = sqrt_2_over_pi();
  auto x3 = elt_multiply(elt_multiply(X, X), X);
  Eigen::Matrix<T, -1, -1> inner = multiply(c, add(X, multiply(0.044715, x3)));
  Eigen::Matrix<T, -1, -1> t = tanh(inner);
  return elt_multiply(multiply(0.5, X), add(1.0, t));
}

// Flax-style LayerNorm: per-row normalize over last axis using biased variance
// (1/N divisor), epsilon = 1e-6, then scale and shift.
template <typename T, typename TG, typename TB>
inline Eigen::Matrix<stan::return_type_t<T, TG, TB>, -1, -1>
layer_norm(const Eigen::Matrix<T, -1, -1>& X,
           const Eigen::Matrix<TG, -1, 1>& gamma,
           const Eigen::Matrix<TB, -1, 1>& beta) {
  using R = stan::return_type_t<T, TG, TB>;
  const int rows = X.rows();
  const int cols = X.cols();
  const double eps = 1e-6;
  const double inv_C = 1.0 / static_cast<double>(cols);
  Eigen::Matrix<R, -1, -1> Y(rows, cols);
  for (int i = 0; i < rows; ++i) {
    Eigen::Matrix<T, -1, 1> row_i = X.row(i).transpose();
    T m = stan::math::sum(row_i) * inv_C;
    Eigen::Matrix<T, -1, 1> centered(cols);
    for (int j = 0; j < cols; ++j) centered(j) = row_i(j) - m;
    Eigen::Matrix<T, -1, 1> sq = stan::math::elt_multiply(centered, centered);
    T v = stan::math::sum(sq) * inv_C;
    T inv_std = stan::math::inv_sqrt(v + eps);
    for (int j = 0; j < cols; ++j) {
      Y(i, j) = centered(j) * inv_std * gamma(j) + beta(j);
    }
  }
  return Y;
}

// Dense layer: X * W + broadcast(b). X (R, K), W (K, Cout), b (Cout,).
template <typename TX, typename TW, typename TB>
inline Eigen::Matrix<stan::return_type_t<TX, TW, TB>, -1, -1>
dense(const Eigen::Matrix<TX, -1, -1>& X,
      const Eigen::Matrix<TW, -1, -1>& W,
      const Eigen::Matrix<TB, -1, 1>& b) {
  using R = stan::return_type_t<TX, TW, TB>;
  Eigen::Matrix<R, -1, -1> Y = stan::math::multiply(X, W);
  // Add bias broadcast over rows
  for (int i = 0; i < Y.rows(); ++i) {
    for (int j = 0; j < Y.cols(); ++j) {
      Y(i, j) = Y(i, j) + b(j);
    }
  }
  return Y;
}

// SpatialGatingUnit(num_heads=1, gate_fn=identity).
// x (L, D_proj_in) is split into z1, z2 each (L, D_gate) where D_gate = D_proj_in/2.
// z2' = LayerNorm(z2); z2'' = sgu_W (L, L) * z2' + bias broadcast over columns.
// out = z1 .* z2''.
template <typename TX, typename TW, typename TB, typename TGS, typename TGB>
inline Eigen::Matrix<stan::return_type_t<TX, TW, TB, TGS, TGB>, -1, -1>
spatial_gating_unit(const Eigen::Matrix<TX, -1, -1>& x,
                    const Eigen::Matrix<TW, -1, -1>& sgu_W,
                    const Eigen::Matrix<TB, -1, 1>& sgu_b,
                    const Eigen::Matrix<TGS, -1, 1>& sgu_norm_scale,
                    const Eigen::Matrix<TGB, -1, 1>& sgu_norm_bias) {
  using R = stan::return_type_t<TX, TW, TB, TGS, TGB>;
  const int L = x.rows();
  const int D = x.cols() / 2;
  Eigen::Matrix<TX, -1, -1> z1 = x.leftCols(D);
  Eigen::Matrix<TX, -1, -1> z2 = x.rightCols(D);
  Eigen::Matrix<R, -1, -1> z2n = layer_norm(z2, sgu_norm_scale, sgu_norm_bias);
  Eigen::Matrix<R, -1, -1> gated = stan::math::multiply(sgu_W, z2n);
  // Add sgu_b[i] to each column j of gated (broadcast over D).
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < D; ++j) {
      gated(i, j) = gated(i, j) + sgu_b(i);
    }
  }
  return stan::math::elt_multiply(z1, gated);
}

}  // namespace gmlp_decode_impl_detail


// Stan's transpiled call site looks like:
//   decode(z, cond, s_mat, embed_W0, embed_b0, embed_W1, embed_b1,
//          ln0_scale, ln0_bias, ln1_scale, ln1_bias, ln2_scale, ln2_bias,
//          sgu_norm_scale, sgu_norm_bias,
//          b0_pi_W0, b0_pi_b0, b0_pi_W1, b0_pi_b1,
//          b0_sgu_W, b0_sgu_b,
//          b0_po_W0, b0_po_b0, b0_po_W1, b0_po_b1,
//          b1_pi_W0, b1_pi_b0, b1_pi_W1, b1_pi_b1,
//          b1_sgu_W, b1_sgu_b,
//          b1_po_W0, b1_po_b0, b1_po_W1, b1_po_b1,
//          head_W0, head_b0, head_W1, head_b1,
//          pstream__);
//
// Each scalar type is templated independently — Stan calls with var for any
// expression that depends on a parameter and double otherwise.

// Stan passes data as Eigen::Map<Matrix>, parameters as plain Eigen::Matrix.
// C++ template arg deduction won't implicitly convert between them, so each
// arg is a generic typename and we materialize to plain Matrix at entry.
template <typename Tz, typename Tc, typename Ts,
          typename T_eW0, typename T_eb0, typename T_eW1, typename T_eb1,
          typename T_ln0s, typename T_ln0b,
          typename T_ln1s, typename T_ln1b,
          typename T_ln2s, typename T_ln2b,
          typename T_sgns, typename T_sgnb,
          typename T0pW0, typename T0pb0, typename T0pW1, typename T0pb1,
          typename T0sW, typename T0sb,
          typename T0qW0, typename T0qb0, typename T0qW1, typename T0qb1,
          typename T1pW0, typename T1pb0, typename T1pW1, typename T1pb1,
          typename T1sW, typename T1sb,
          typename T1qW0, typename T1qb0, typename T1qW1, typename T1qb1,
          typename T_hW0, typename T_hb0, typename T_hW1, typename T_hb1>
inline Eigen::Matrix<stan::return_type_t<
    typename Tz::Scalar, typename Tc::Scalar, typename Ts::Scalar,
    typename T_eW0::Scalar, typename T_eb0::Scalar,
    typename T_eW1::Scalar, typename T_eb1::Scalar,
    typename T_ln0s::Scalar, typename T_ln0b::Scalar,
    typename T_ln1s::Scalar, typename T_ln1b::Scalar,
    typename T_ln2s::Scalar, typename T_ln2b::Scalar,
    typename T_sgns::Scalar, typename T_sgnb::Scalar,
    typename T0pW0::Scalar, typename T0pb0::Scalar,
    typename T0pW1::Scalar, typename T0pb1::Scalar,
    typename T0sW::Scalar, typename T0sb::Scalar,
    typename T0qW0::Scalar, typename T0qb0::Scalar,
    typename T0qW1::Scalar, typename T0qb1::Scalar,
    typename T1pW0::Scalar, typename T1pb0::Scalar,
    typename T1pW1::Scalar, typename T1pb1::Scalar,
    typename T1sW::Scalar, typename T1sb::Scalar,
    typename T1qW0::Scalar, typename T1qb0::Scalar,
    typename T1qW1::Scalar, typename T1qb1::Scalar,
    typename T_hW0::Scalar, typename T_hb0::Scalar,
    typename T_hW1::Scalar, typename T_hb1::Scalar>, -1, 1>
decode(const Tz& z_in, const Tc& cond_in, const Ts& s_mat_in,
       const T_eW0& embed_W0_in, const T_eb0& embed_b0_in,
       const T_eW1& embed_W1_in, const T_eb1& embed_b1_in,
       const T_ln0s& ln0_scale_in, const T_ln0b& ln0_bias_in,
       const T_ln1s& ln1_scale_in, const T_ln1b& ln1_bias_in,
       const T_ln2s& ln2_scale_in, const T_ln2b& ln2_bias_in,
       const T_sgns& sgu_norm_scale_in, const T_sgnb& sgu_norm_bias_in,
       const T0pW0& b0_pi_W0_in, const T0pb0& b0_pi_b0_in,
       const T0pW1& b0_pi_W1_in, const T0pb1& b0_pi_b1_in,
       const T0sW&  b0_sgu_W_in, const T0sb&  b0_sgu_b_in,
       const T0qW0& b0_po_W0_in, const T0qb0& b0_po_b0_in,
       const T0qW1& b0_po_W1_in, const T0qb1& b0_po_b1_in,
       const T1pW0& b1_pi_W0_in, const T1pb0& b1_pi_b0_in,
       const T1pW1& b1_pi_W1_in, const T1pb1& b1_pi_b1_in,
       const T1sW&  b1_sgu_W_in, const T1sb&  b1_sgu_b_in,
       const T1qW0& b1_po_W0_in, const T1qb0& b1_po_b0_in,
       const T1qW1& b1_po_W1_in, const T1qb1& b1_po_b1_in,
       const T_hW0& head_W0_in, const T_hb0& head_b0_in,
       const T_hW1& head_W1_in, const T_hb1& head_b1_in,
       std::ostream* pstream__) {
  using namespace gmlp_decode_impl_detail;
  using stan::math::multiply;

  // Materialize each arg to a plain Matrix so the helpers below can take
  // concrete Eigen::Matrix<Scalar, ..., ...> by const ref. Implicit conversion
  // from Eigen::Map to Eigen::Matrix is fine; it's just a copy.
  Eigen::Matrix<typename Tz::Scalar, -1, 1> z = z_in;
  Eigen::Matrix<typename Tc::Scalar, -1, 1> cond = cond_in;
  Eigen::Matrix<typename Ts::Scalar, -1, -1> s_mat = s_mat_in;
  Eigen::Matrix<typename T_eW0::Scalar, -1, -1> embed_W0 = embed_W0_in;
  Eigen::Matrix<typename T_eb0::Scalar, -1, 1>  embed_b0 = embed_b0_in;
  Eigen::Matrix<typename T_eW1::Scalar, -1, -1> embed_W1 = embed_W1_in;
  Eigen::Matrix<typename T_eb1::Scalar, -1, 1>  embed_b1 = embed_b1_in;
  Eigen::Matrix<typename T_ln0s::Scalar, -1, 1> ln0_scale = ln0_scale_in;
  Eigen::Matrix<typename T_ln0b::Scalar, -1, 1> ln0_bias  = ln0_bias_in;
  Eigen::Matrix<typename T_ln1s::Scalar, -1, 1> ln1_scale = ln1_scale_in;
  Eigen::Matrix<typename T_ln1b::Scalar, -1, 1> ln1_bias  = ln1_bias_in;
  Eigen::Matrix<typename T_ln2s::Scalar, -1, 1> ln2_scale = ln2_scale_in;
  Eigen::Matrix<typename T_ln2b::Scalar, -1, 1> ln2_bias  = ln2_bias_in;
  Eigen::Matrix<typename T_sgns::Scalar, -1, 1> sgu_norm_scale = sgu_norm_scale_in;
  Eigen::Matrix<typename T_sgnb::Scalar, -1, 1> sgu_norm_bias  = sgu_norm_bias_in;
  Eigen::Matrix<typename T0pW0::Scalar, -1, -1> b0_pi_W0 = b0_pi_W0_in;
  Eigen::Matrix<typename T0pb0::Scalar, -1, 1>  b0_pi_b0 = b0_pi_b0_in;
  Eigen::Matrix<typename T0pW1::Scalar, -1, -1> b0_pi_W1 = b0_pi_W1_in;
  Eigen::Matrix<typename T0pb1::Scalar, -1, 1>  b0_pi_b1 = b0_pi_b1_in;
  Eigen::Matrix<typename T0sW::Scalar,  -1, -1> b0_sgu_W = b0_sgu_W_in;
  Eigen::Matrix<typename T0sb::Scalar,  -1, 1>  b0_sgu_b = b0_sgu_b_in;
  Eigen::Matrix<typename T0qW0::Scalar, -1, -1> b0_po_W0 = b0_po_W0_in;
  Eigen::Matrix<typename T0qb0::Scalar, -1, 1>  b0_po_b0 = b0_po_b0_in;
  Eigen::Matrix<typename T0qW1::Scalar, -1, -1> b0_po_W1 = b0_po_W1_in;
  Eigen::Matrix<typename T0qb1::Scalar, -1, 1>  b0_po_b1 = b0_po_b1_in;
  Eigen::Matrix<typename T1pW0::Scalar, -1, -1> b1_pi_W0 = b1_pi_W0_in;
  Eigen::Matrix<typename T1pb0::Scalar, -1, 1>  b1_pi_b0 = b1_pi_b0_in;
  Eigen::Matrix<typename T1pW1::Scalar, -1, -1> b1_pi_W1 = b1_pi_W1_in;
  Eigen::Matrix<typename T1pb1::Scalar, -1, 1>  b1_pi_b1 = b1_pi_b1_in;
  Eigen::Matrix<typename T1sW::Scalar,  -1, -1> b1_sgu_W = b1_sgu_W_in;
  Eigen::Matrix<typename T1sb::Scalar,  -1, 1>  b1_sgu_b = b1_sgu_b_in;
  Eigen::Matrix<typename T1qW0::Scalar, -1, -1> b1_po_W0 = b1_po_W0_in;
  Eigen::Matrix<typename T1qb0::Scalar, -1, 1>  b1_po_b0 = b1_po_b0_in;
  Eigen::Matrix<typename T1qW1::Scalar, -1, -1> b1_po_W1 = b1_po_W1_in;
  Eigen::Matrix<typename T1qb1::Scalar, -1, 1>  b1_po_b1 = b1_po_b1_in;
  Eigen::Matrix<typename T_hW0::Scalar, -1, -1> head_W0  = head_W0_in;
  Eigen::Matrix<typename T_hb0::Scalar, -1, 1>  head_b0  = head_b0_in;
  Eigen::Matrix<typename T_hW1::Scalar, -1, -1> head_W1  = head_W1_in;
  Eigen::Matrix<typename T_hb1::Scalar, -1, 1>  head_b1  = head_b1_in;

  using R = stan::return_type_t<
      typename Tz::Scalar, typename Tc::Scalar, typename Ts::Scalar,
      typename T_eW0::Scalar, typename T_eb0::Scalar,
      typename T_eW1::Scalar, typename T_eb1::Scalar,
      typename T_ln0s::Scalar, typename T_ln0b::Scalar,
      typename T_ln1s::Scalar, typename T_ln1b::Scalar,
      typename T_ln2s::Scalar, typename T_ln2b::Scalar,
      typename T_sgns::Scalar, typename T_sgnb::Scalar,
      typename T0pW0::Scalar, typename T0pb0::Scalar,
      typename T0pW1::Scalar, typename T0pb1::Scalar,
      typename T0sW::Scalar, typename T0sb::Scalar,
      typename T0qW0::Scalar, typename T0qb0::Scalar,
      typename T0qW1::Scalar, typename T0qb1::Scalar,
      typename T1pW0::Scalar, typename T1pb0::Scalar,
      typename T1pW1::Scalar, typename T1pb1::Scalar,
      typename T1sW::Scalar, typename T1sb::Scalar,
      typename T1qW0::Scalar, typename T1qb0::Scalar,
      typename T1qW1::Scalar, typename T1qb1::Scalar,
      typename T_hW0::Scalar, typename T_hb0::Scalar,
      typename T_hW1::Scalar, typename T_hb1::Scalar>;

  const int L = z.size();
  const int C = cond.size();
  const int F_IN = 1 + 2 + C;

  // Build input features x[L, F_IN] = [z, s_mat, cond_broadcast].
  Eigen::Matrix<R, -1, -1> x(L, F_IN);
  for (int i = 0; i < L; ++i) {
    x(i, 0) = z(i);
    x(i, 1) = s_mat(i, 0);
    x(i, 2) = s_mat(i, 1);
    for (int k = 0; k < C; ++k) x(i, 3 + k) = cond(k);
  }

  // embed: Dense(64) -> gelu -> Dense(64). Final layer is linear in MLP.
  Eigen::Matrix<R, -1, -1> e = gelu_m(dense(x, embed_W0, embed_b0));
  e = dense(e, embed_W1, embed_b1);

  // Block 0
  {
    auto y0 = layer_norm(e, ln0_scale, ln0_bias);
    auto pi0 = dense(gelu_m(dense(y0, b0_pi_W0, b0_pi_b0)),
                     b0_pi_W1, b0_pi_b1);
    auto g0 = spatial_gating_unit(pi0, b0_sgu_W, b0_sgu_b,
                                  sgu_norm_scale, sgu_norm_bias);
    auto po0 = dense(gelu_m(dense(g0, b0_po_W0, b0_po_b0)),
                     b0_po_W1, b0_po_b1);
    e = stan::math::add(e, po0);
  }

  // Block 1
  {
    auto y1 = layer_norm(e, ln1_scale, ln1_bias);
    auto pi1 = dense(gelu_m(dense(y1, b1_pi_W0, b1_pi_b0)),
                     b1_pi_W1, b1_pi_b1);
    auto g1 = spatial_gating_unit(pi1, b1_sgu_W, b1_sgu_b,
                                  sgu_norm_scale, sgu_norm_bias);
    auto po1 = dense(gelu_m(dense(g1, b1_po_W0, b1_po_b0)),
                     b1_po_W1, b1_po_b1);
    e = stan::math::add(e, po1);
  }

  // Head
  auto e_n = layer_norm(e, ln2_scale, ln2_bias);
  auto h = gelu_m(dense(e_n, head_W0, head_b0));
  auto out_mat = dense(h, head_W1, head_b1);

  Eigen::Matrix<R, -1, 1> out(L);
  for (int i = 0; i < L; ++i) out(i) = out_mat(i, 0);
  return out;
}

#endif  // GMLP_DECODE_IMPL_HPP
