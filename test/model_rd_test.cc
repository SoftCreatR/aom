/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <cstdlib>

#include "gtest/gtest.h"
#include "config/av1_rtcd.h"

#include "test/acm_random.h"
#include "test/register_state_check.h"
#include "test/util.h"

using libaom_test::ACMRandom;

namespace {

using InterpCubicRateDistFunc = void (*)(const double *p1, const double *p2,
                                         double x, double *const rate_f,
                                         double *const distbysse_f);

using InterpCubicTestParam = std::tuple<const InterpCubicRateDistFunc>;

class InterpCubicTest : public ::testing::TestWithParam<InterpCubicTestParam> {
 public:
  double generate_random_double(double min, double max) {
    return min + (static_cast<double>(rnd_.Rand31()) / ((1U << 31) - 1)) *
                     (max - min);
  }
  void SetUp() override { target_func_ = GET_PARAM(0); }
  void TearDown() override {}
  void CheckOutput();
  void SpeedTest();

 protected:
  InterpCubicRateDistFunc target_func_;

 private:
  libaom_test::ACMRandom rnd_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(InterpCubicTest);

void InterpCubicTest::CheckOutput() {
  double p1[4], p2[4];
  double rate_f_ref, rate_f_mod, distbysse_f_ref, distbysse_f_mod;
  const int knum_iter = 10000;
  for (int iter = 0; iter < knum_iter; iter++) {
    for (int i = 0; i < 4; i++) {
      p1[i] = generate_random_double(0.0000, 4096.000000);
      p2[i] = generate_random_double(0.0000, 16.0000);
    }
    double x = generate_random_double(0.0000, 1.0000);

    av1_interp_cubic_rate_dist_c(p1, p2, x, &rate_f_ref, &distbysse_f_ref);
    target_func_(p1, p2, x, &rate_f_mod, &distbysse_f_mod);
    EXPECT_EQ(rate_f_ref, rate_f_mod) << "Error: rate_f value mismatch";
    EXPECT_EQ(distbysse_f_ref, distbysse_f_mod)
        << "Error: distbysse_f value mismatch";
  }
}

void InterpCubicTest::SpeedTest() {
  double p1[4], p2[4];
  double rate_f_ref, rate_f_mod, distbysse_f_ref, distbysse_f_mod;

  for (int i = 0; i < 4; i++) {
    p1[i] = generate_random_double(0.0000, 4096.0000);
    p2[i] = generate_random_double(0.0000, 16.0000);
  }
  double x = generate_random_double(0.0000, 1.0000);

  const int num_iter = 100000000;

  aom_usec_timer ref_timer, test_timer;
  aom_usec_timer_start(&ref_timer);
  for (int iter = 0; iter < num_iter; iter++) {
    av1_interp_cubic_rate_dist_c(p1, p2, x, &rate_f_ref, &distbysse_f_ref);
  }
  aom_usec_timer_mark(&ref_timer);
  const int elapsed_time_c =
      static_cast<int>(aom_usec_timer_elapsed(&ref_timer));

  aom_usec_timer_start(&test_timer);
  for (int iter = 0; iter < num_iter; iter++) {
    target_func_(p1, p2, x, &rate_f_mod, &distbysse_f_mod);
  }
  aom_usec_timer_mark(&test_timer);
  const int elapsed_time_simd =
      static_cast<int>(aom_usec_timer_elapsed(&test_timer));

  printf(
      " c_time=%d \t simd_time=%d \t "
      "Scaling=%lf \n",
      elapsed_time_c, elapsed_time_simd,
      (static_cast<double>(elapsed_time_c) / elapsed_time_simd));
}

TEST_P(InterpCubicTest, CheckOutput) { CheckOutput(); }

TEST_P(InterpCubicTest, DISABLED_Speed) { SpeedTest(); }

#if HAVE_SSE4_1
INSTANTIATE_TEST_SUITE_P(SSE4_1, InterpCubicTest,
                         ::testing::Values(av1_interp_cubic_rate_dist_sse4_1));
#endif  // HAVE_SSE4_1

}  // namespace
