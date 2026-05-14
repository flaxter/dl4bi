// External C++ implementation of MLPDeepRV(dims=[L, L]).decode, linked into
// the Stan model via --allow-undefined. Same pattern as gmlp_decode_impl.hpp
// but for the simpler MLP forward (concat -> Dense -> relu -> Dense).
//
// The all-Stan port in mlp_decode.stan uses a scalar loop for ReLU and
// for-loop-based bias broadcasts. The wins from doing this externally
// are bounded for MLP because:
//   1. The two matmuls already dispatch to stan::math::multiply's
//      optimized matrix-AD overload in both versions.
//   2. ReLU on a length-L vector creates only ~L tape entries, dwarfed
//      by the matmul work.
// Realistic expectation: 1.2-1.5x speedup, much less than gMLP.

#ifndef MLP_DECODE_IMPL_HPP
#define MLP_DECODE_IMPL_HPP

#include <stan/math.hpp>
#include <Eigen/Dense>
#include <ostream>

// ReLU written algebraically: 0.5 * (x + |x|).
// Routes through vectorized stan::math::abs and stan::math::add overloads.
template <typename T>
inline Eigen::Matrix<T, -1, 1> relu_v(const Eigen::Matrix<T, -1, 1>& x) {
  return stan::math::multiply(0.5,
                              stan::math::add(x, stan::math::abs(x)));
}

// Stan passes data as Eigen::Map, parameters as plain Eigen::Matrix. Use
// generic typenames so template arg deduction accepts both.
template <typename Tz, typename Tc,
          typename TW1, typename Tb1,
          typename TW2, typename Tb2>
inline Eigen::Matrix<stan::return_type_t<
    typename Tz::Scalar, typename Tc::Scalar,
    typename TW1::Scalar, typename Tb1::Scalar,
    typename TW2::Scalar, typename Tb2::Scalar>, -1, 1>
decode(const Tz& z_in, const Tc& cond_in,
       const TW1& W1_in, const Tb1& b1_in,
       const TW2& W2_in, const Tb2& b2_in,
       std::ostream* pstream__) {
  using R = stan::return_type_t<
      typename Tz::Scalar, typename Tc::Scalar,
      typename TW1::Scalar, typename Tb1::Scalar,
      typename TW2::Scalar, typename Tb2::Scalar>;

  // Materialize to plain Matrix so the helpers below take concrete types.
  Eigen::Matrix<typename Tz::Scalar, -1, 1>   z   = z_in;
  Eigen::Matrix<typename Tc::Scalar, -1, 1>   c   = cond_in;
  Eigen::Matrix<typename TW1::Scalar, -1, -1> W1 = W1_in;
  Eigen::Matrix<typename Tb1::Scalar, -1, 1>  b1 = b1_in;
  Eigen::Matrix<typename TW2::Scalar, -1, -1> W2 = W2_in;
  Eigen::Matrix<typename Tb2::Scalar, -1, 1>  b2 = b2_in;

  const int L = z.size();
  const int C = c.size();

  // x = [z; cond]   (length L + C)
  Eigen::Matrix<R, -1, 1> x(L + C);
  for (int i = 0; i < L; ++i) x(i)     = z(i);
  for (int j = 0; j < C; ++j) x(L + j) = c(j);

  // h = relu(W1^T x + b1)
  Eigen::Matrix<R, -1, 1> h1 = stan::math::add(
      stan::math::multiply(stan::math::transpose(W1), x), b1);
  Eigen::Matrix<R, -1, 1> h = relu_v(h1);

  // out = W2^T h + b2
  return stan::math::add(
      stan::math::multiply(stan::math::transpose(W2), h), b2);
}

#endif  // MLP_DECODE_IMPL_HPP
