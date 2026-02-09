/*
 *  Copyright (c) 2025 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"

DECLARE_ALIGNED(64, static const uint16_t,
                rshift_1w[32]) = { 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                   12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 29, 30, 31, 31 };
DECLARE_ALIGNED(64, static const uint16_t,
                rshift_2w[32]) = { 2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                   24, 25, 26, 27, 28, 29, 30, 31, 31, 31 };

static INLINE __m512i avg3_epu16_avx512(const __m512i *x, const __m512i *y,
                                        const __m512i *z) {
  const __m512i one = _mm512_set1_epi16(1);
  const __m512i a = _mm512_avg_epu16(*x, *z);
  const __m512i b =
      _mm512_subs_epu16(a, _mm512_and_si512(_mm512_xor_si512(*x, *z), one));
  return _mm512_avg_epu16(b, *y);
}

#define DC_STORE_SUM_32(offset, shift)                                    \
  do {                                                                    \
    __m256i sum256 = _mm256_add_epi16(_mm512_castsi512_si256(sum),        \
                                      _mm512_extracti64x4_epi64(sum, 1)); \
    __m128i sum128 = _mm_add_epi16(_mm256_castsi256_si128(sum256),        \
                                   _mm256_extracti128_si256(sum256, 1));  \
    sum128 = _mm_cvtepu16_epi32(_mm_hadd_epi16(sum128, sum128));          \
    sum128 = _mm_hadd_epi32(sum128, sum128);                              \
    sum128 = _mm_hadd_epi32(sum128, sum128);                              \
    sum128 = _mm_add_epi32(sum128, _mm_cvtsi32_si128(offset));            \
    sum128 = _mm_srli_epi32(sum128, shift);                               \
    __m512i v = _mm512_broadcastw_epi16(sum128);                          \
    for (int i = 0; i < 32; ++i) {                                        \
      _mm512_store_si512((__m512i *)dst, v);                              \
      dst += stride;                                                      \
    }                                                                     \
  } while (0)

void vpx_highbd_dc_predictor_32x32_avx512(uint16_t *dst, ptrdiff_t stride,
                                          const uint16_t *above,
                                          const uint16_t *left, int bd) {
  (void)bd;
  __m512i sum =
      _mm512_add_epi16(*(const __m512i *)above, *(const __m512i *)left);
  DC_STORE_SUM_32(32, 6);
}

void vpx_highbd_dc_left_predictor_32x32_avx512(uint16_t *dst, ptrdiff_t stride,
                                               const uint16_t *above,
                                               const uint16_t *left, int bd) {
  (void)above;
  (void)bd;
  __m512i sum = _mm512_load_si512((const __m512i *)left);
  DC_STORE_SUM_32(16, 5);
}

void vpx_highbd_dc_top_predictor_32x32_avx512(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *above,
                                              const uint16_t *left, int bd) {
  (void)left;
  (void)bd;
  __m512i sum = _mm512_load_si512((const __m512i *)above);
  DC_STORE_SUM_32(16, 5);
}

void vpx_highbd_dc_128_predictor_32x32_avx512(uint16_t *dst, ptrdiff_t stride,
                                              const uint16_t *above,
                                              const uint16_t *left, int bd) {
  (void)above;
  (void)left;
  const __m512i v = _mm512_set1_epi16(1 << (bd - 1));
  for (int i = 0; i < 32; ++i) {
    _mm512_store_si512((__m512i *)dst, v);
    dst += stride;
  }
}

void vpx_highbd_d63_predictor_32x32_avx512(uint16_t *dst, ptrdiff_t stride,
                                           const uint16_t *above,
                                           const uint16_t *left, int bd) {
  (void)left;
  (void)bd;

  const __m512i A = _mm512_load_si512((const __m512i *)above);

  // B = shift by 1 (2 bytes)
  const __m512i B = _mm512_permutexvar_epi16(*(const __m512i *)rshift_1w, A);

  // C = shift by 2 (4 bytes)
  const __m512i C = _mm512_permutexvar_epi16(*(const __m512i *)rshift_2w, A);

  __m512i avg2 = _mm512_avg_epu16(A, B);
  __m512i avg3 = avg3_epu16_avx512(&A, &B, &C);

  for (int i = 0; i < 30; i += 2) {
    _mm512_store_si512((__m512i *)dst, avg2);
    dst += stride;

    _mm512_store_si512((__m512i *)dst, avg3);
    dst += stride;

    avg2 = _mm512_permutexvar_epi16(*(const __m512i *)rshift_1w, avg2);
    avg3 = _mm512_permutexvar_epi16(*(const __m512i *)rshift_1w, avg3);
  }

  _mm512_store_si512((__m512i *)dst, avg2);
  dst += stride;

  _mm512_store_si512((__m512i *)dst, avg3);
}

static INLINE void d207_store_16x32_avx512(uint16_t **dst,
                                           const ptrdiff_t stride,
                                           const __m512i *abcd,
                                           const __m512i *efgh) {
  _mm512_store_si512((__m512i *)*dst, *abcd);
  *dst += stride;

#define D207_STORE_ROW(shift)                               \
  do {                                                      \
    __m512i row = _mm512_alignr_epi32(*efgh, *abcd, shift); \
    _mm512_store_si512((__m512i *)*dst, row);               \
    *dst += stride;                                         \
  } while (0)

  D207_STORE_ROW(1);
  D207_STORE_ROW(2);
  D207_STORE_ROW(3);
  D207_STORE_ROW(4);
  D207_STORE_ROW(5);
  D207_STORE_ROW(6);
  D207_STORE_ROW(7);
  D207_STORE_ROW(8);
  D207_STORE_ROW(9);
  D207_STORE_ROW(10);
  D207_STORE_ROW(11);
  D207_STORE_ROW(12);
  D207_STORE_ROW(13);
  D207_STORE_ROW(14);
  D207_STORE_ROW(15);
#undef D207_STORE_ROW
}

void vpx_highbd_d207_predictor_32x32_avx512(uint16_t *dst, ptrdiff_t stride,
                                            const uint16_t *above,
                                            const uint16_t *left, int bd) {
  (void)above;
  (void)bd;

  const __m512i A = _mm512_load_si512((const __m512i *)left);
  const __m512i AR = _mm512_set1_epi16(left[31]);

  // B = shift by 1 (2 bytes)
  __m512i B = _mm512_permutexvar_epi16(*(const __m512i *)rshift_1w, A);
  // C = shift by 2 (4 bytes)
  __m512i C = _mm512_alignr_epi32(AR, A, 1);

  __m512i avg2 = _mm512_avg_epu16(A, B);
  __m512i avg3 = avg3_epu16_avx512(&A, &B, &C);

  __m512i out_aceg = _mm512_unpacklo_epi16(avg2, avg3);
  __m512i out_bdfh = _mm512_unpackhi_epi16(avg2, avg3);

  __m512i out_abcd = _mm512_permutex2var_epi64(
      out_aceg, _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0), out_bdfh);
  __m512i out_efgh = _mm512_permutex2var_epi64(
      out_aceg, _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4), out_bdfh);

  d207_store_16x32_avx512(&dst, stride, &out_abcd, &out_efgh);
  d207_store_16x32_avx512(&dst, stride, &out_efgh, &AR);
}

void vpx_highbd_v_predictor_32x32_avx512(uint16_t *dst, ptrdiff_t stride,
                                         const uint16_t *above,
                                         const uint16_t *left, int bd) {
  (void)left;
  (void)bd;
  __m512i A = _mm512_load_si512((const __m512i *)above);
  for (int i = 0; i < 32; ++i) {
    _mm512_store_si512((__m512i *)dst, A);
    dst += stride;
  }
}

static VPX_FORCE_INLINE void h_store_8x32_avx512(uint16_t **dst,
                                                 const ptrdiff_t stride,
                                                 const __m128i l) {
  __m512i row[8];
  row[0] = _mm512_broadcastw_epi16(l);
  row[1] = _mm512_broadcastw_epi16(_mm_srli_si128(l, 2));
  row[2] = _mm512_broadcastw_epi16(_mm_srli_si128(l, 4));
  row[3] = _mm512_broadcastw_epi16(_mm_srli_si128(l, 6));
  row[4] = _mm512_broadcastw_epi16(_mm_srli_si128(l, 8));
  row[5] = _mm512_broadcastw_epi16(_mm_srli_si128(l, 10));
  row[6] = _mm512_broadcastw_epi16(_mm_srli_si128(l, 12));
  row[7] = _mm512_broadcastw_epi16(_mm_srli_si128(l, 14));

  for (int i = 0; i < 8; ++i) {
    _mm512_store_si512((__m512i *)*dst, row[i]);
    *dst += stride;
  }
}

void vpx_highbd_h_predictor_32x32_avx512(uint16_t *dst, ptrdiff_t stride,
                                         const uint16_t *above,
                                         const uint16_t *left, int bd) {
  (void)above;
  (void)bd;
  __m512i L512 = _mm512_load_si512((const __m512i *)left);

  __m128i L = _mm512_castsi512_si128(L512);
  h_store_8x32_avx512(&dst, stride, L);

  L = _mm512_extracti32x4_epi32(L512, 1);
  h_store_8x32_avx512(&dst, stride, L);

  L = _mm512_extracti32x4_epi32(L512, 2);
  h_store_8x32_avx512(&dst, stride, L);

  L = _mm512_extracti32x4_epi32(L512, 3);
  h_store_8x32_avx512(&dst, stride, L);
}

void vpx_highbd_tm_predictor_32x32_avx512(uint16_t *dst, ptrdiff_t stride,
                                          const uint16_t *above,
                                          const uint16_t *left, int bd) {
  __m512i top_left = _mm512_set1_epi16(above[-1]);
  __m512i A = _mm512_sub_epi16(*(const __m512i *)above, top_left);

  __m512i bd_max = _mm512_set1_epi16((1 << bd) - 1);
  __m512i bd_min = _mm512_setzero_si512();

  for (int i = 0; i < 16; i++) {
    __m512i L0 = _mm512_set1_epi16(left[0]);
    __m512i L1 = _mm512_set1_epi16(left[1]);

    __m512i D = _mm512_add_epi16(A, L0);

    D = _mm512_min_epi16(D, bd_max);
    D = _mm512_max_epi16(D, bd_min);

    _mm512_store_si512((__m512i *)dst, D);

    D = _mm512_add_epi16(A, L1);

    D = _mm512_min_epi16(D, bd_max);
    D = _mm512_max_epi16(D, bd_min);

    _mm512_store_si512((__m512i *)(dst + stride), D);

    dst += 2 * stride;
    left += 2;
  }
}
