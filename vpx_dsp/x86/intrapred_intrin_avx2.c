/*
 *  Copyright (c) 2026 The WebM project authors. All Rights Reserved.
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

static INLINE __m256i avg3_epu8_avx2(const __m256i *x, const __m256i *y,
                                     const __m256i *z) {
  const __m256i one = _mm256_set1_epi8(1);
  const __m256i a = _mm256_avg_epu8(*x, *z);
  const __m256i b =
      _mm256_subs_epu8(a, _mm256_and_si256(_mm256_xor_si256(*x, *z), one));
  return _mm256_avg_epu8(b, *y);
}

void vpx_dc_predictor_32x32_avx2(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  __m256i zero = _mm256_setzero_si256();

  __m256i A = _mm256_load_si256((const __m256i *)above);
  __m256i L = _mm256_load_si256((const __m256i *)left);

  __m256i sumA = _mm256_sad_epu8(A, zero);
  __m256i sumL = _mm256_sad_epu8(L, zero);
  __m256i sum = _mm256_add_epi16(sumA, sumL);

  __m128i sum128 = _mm_add_epi16(_mm256_castsi256_si128(sum),
                                 _mm256_extracti128_si256(sum, 1));
  sum128 = _mm_add_epi16(sum128, _mm_unpackhi_epi64(sum128, sum128));
  sum128 = _mm_add_epi16(sum128, _mm_cvtsi32_si128(32));
  sum128 = _mm_srli_epi16(sum128, 6);

  __m256i v = _mm256_broadcastb_epi8(sum128);

  for (int i = 0; i < 32; ++i) {
    _mm256_store_si256((__m256i *)dst, v);
    dst += stride;
  }
}

void vpx_d63_predictor_32x32_avx2(uint8_t *dst, ptrdiff_t stride,
                                  const uint8_t *above, const uint8_t *left) {
  (void)left;
  const __m256i A = _mm256_load_si256((const __m256i *)above);
  const __m256i AR = _mm256_set1_epi8(above[31]);

  __m256i ar_a = _mm256_permute2x128_si256(AR, A, 3);
  __m256i B = _mm256_alignr_epi8(ar_a, A, 1);
  __m256i C = _mm256_alignr_epi8(ar_a, A, 2);

  __m256i avg2 = _mm256_avg_epu8(A, B);
  __m256i avg3 = avg3_epu8_avx2(&A, &B, &C);

  __m256i ar_avg2 = _mm256_permute2x128_si256(AR, avg2, 3);
  __m256i ar_avg3 = _mm256_permute2x128_si256(AR, avg3, 3);

  __m256i row0 = avg2;
  __m256i row1 = avg3;
  _mm256_store_si256((__m256i *)dst, row0);
  dst += stride;
  _mm256_store_si256((__m256i *)dst, row1);
  dst += stride;

#define D63_STORE_2x32(i)                          \
  do {                                             \
    row0 = _mm256_alignr_epi8(ar_avg2, avg2, (i)); \
    row1 = _mm256_alignr_epi8(ar_avg3, avg3, (i)); \
    _mm256_store_si256((__m256i *)dst, row0);      \
    dst += stride;                                 \
    _mm256_store_si256((__m256i *)dst, row1);      \
    dst += stride;                                 \
  } while (0)

  D63_STORE_2x32(1);
  D63_STORE_2x32(2);
  D63_STORE_2x32(3);
  D63_STORE_2x32(4);
  D63_STORE_2x32(5);
  D63_STORE_2x32(6);
  D63_STORE_2x32(7);
  D63_STORE_2x32(8);
  D63_STORE_2x32(9);
  D63_STORE_2x32(10);
  D63_STORE_2x32(11);
  D63_STORE_2x32(12);
  D63_STORE_2x32(13);
  D63_STORE_2x32(14);
  D63_STORE_2x32(15);
#undef D63_STORE_2x32
}

static INLINE void d207_store_16x32_avx2(uint8_t **dst, const ptrdiff_t stride,
                                         const __m256i *low,
                                         const __m256i *high) {
  _mm256_store_si256((__m256i *)*dst, *low);
  *dst += stride;

  __m256i shift;
  __m256i mid = _mm256_permute2x128_si256(*high, *low, 3);

#define D207_STORE_32(h, l, i)                  \
  do {                                          \
    shift = _mm256_alignr_epi8((h), (l), (i));  \
    _mm256_store_si256((__m256i *)*dst, shift); \
    *dst += stride;                             \
  } while (0)

  D207_STORE_32(mid, *low, 2);
  D207_STORE_32(mid, *low, 4);
  D207_STORE_32(mid, *low, 6);
  D207_STORE_32(mid, *low, 8);
  D207_STORE_32(mid, *low, 10);
  D207_STORE_32(mid, *low, 12);
  D207_STORE_32(mid, *low, 14);

  _mm256_store_si256((__m256i *)*dst, mid);
  *dst += stride;

  D207_STORE_32(*high, mid, 2);
  D207_STORE_32(*high, mid, 4);
  D207_STORE_32(*high, mid, 6);
  D207_STORE_32(*high, mid, 8);
  D207_STORE_32(*high, mid, 10);
  D207_STORE_32(*high, mid, 12);
  D207_STORE_32(*high, mid, 14);
#undef D207_STORE_32
}

void vpx_d207_predictor_32x32_avx2(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above, const uint8_t *left) {
  (void)above;
  const __m256i L = _mm256_load_si256((const __m256i *)left);
  const __m256i LR = _mm256_set1_epi8(left[31]);

  __m256i lr_l = _mm256_permute2x128_si256(LR, L, 3);

  __m256i B = _mm256_alignr_epi8(lr_l, L, 1);
  __m256i C = _mm256_alignr_epi8(lr_l, L, 2);

  __m256i avg2 = _mm256_avg_epu8(L, B);
  __m256i avg3 = avg3_epu8_avx2(&L, &B, &C);

  __m256i out_ac = _mm256_unpacklo_epi8(avg2, avg3);
  __m256i out_bd = _mm256_unpackhi_epi8(avg2, avg3);

  __m256i out_ab = _mm256_permute2x128_si256(out_ac, out_bd, 0x20);
  __m256i out_cd = _mm256_permute2x128_si256(out_ac, out_bd, 0x31);

  d207_store_16x32_avx2(&dst, stride, &out_ab, &out_cd);
  d207_store_16x32_avx2(&dst, stride, &out_cd, &LR);
}

void vpx_v_predictor_32x32_avx2(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  (void)left;
  __m256i A = _mm256_load_si256((const __m256i *)above);
  for (int i = 0; i < 32; ++i) {
    _mm256_store_si256((__m256i *)dst, A);
    dst += stride;
  }
}

void vpx_h_predictor_32x32_avx2(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  (void)above;
  __m256i L256 = _mm256_load_si256((const __m256i *)left);

  __m128i L = _mm256_castsi256_si128(L256);
  for (int i = 0; i < 16; ++i) {
    _mm256_store_si256((__m256i *)dst, _mm256_broadcastb_epi8(L));
    dst += stride;
    L = _mm_srli_si128(L, 1);
  }

  L = _mm256_extracti128_si256(L256, 1);
  for (int i = 0; i < 16; ++i) {
    _mm256_store_si256((__m256i *)dst, _mm256_broadcastb_epi8(L));
    dst += stride;
    L = _mm_srli_si128(L, 1);
  }
}

void vpx_tm_predictor_8x8_avx2(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  __m128i top_left = _mm_set1_epi16(above[-1]);
  __m128i A = _mm_sub_epi16(_mm_cvtepu8_epi16(_mm_loadu_si64(above)), top_left);

  for (int i = 0; i < 4; i++) {
    __m128i L0 = _mm_set1_epi16(left[0]);
    __m128i D = _mm_add_epi16(A, L0);
    __m128i packed = _mm_packus_epi16(D, _mm_undefined_si128());
    _mm_storeu_si64(dst, packed);

    __m128i L1 = _mm_set1_epi16(left[1]);
    D = _mm_add_epi16(A, L1);
    packed = _mm_packus_epi16(D, _mm_undefined_si128());
    _mm_storeu_si64(dst + stride, packed);
    dst += 2 * stride;
    left += 2;
  }
}

void vpx_tm_predictor_16x16_avx2(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  __m256i top_left = _mm256_set1_epi16(above[-1]);
  __m256i A =
      _mm256_sub_epi16(_mm256_cvtepu8_epi16(*(const __m128i *)above), top_left);

  for (int i = 0; i < 8; i++) {
    __m256i L0 = _mm256_set1_epi16(left[0]);
    __m256i D = _mm256_add_epi16(A, L0);
    __m128i packed = _mm_packus_epi16(_mm256_castsi256_si128(D),
                                      _mm256_extracti128_si256(D, 1));
    _mm_store_si128((__m128i *)dst, packed);

    __m256i L1 = _mm256_set1_epi16(left[1]);
    D = _mm256_add_epi16(A, L1);
    packed = _mm_packus_epi16(_mm256_castsi256_si128(D),
                              _mm256_extracti128_si256(D, 1));
    _mm_store_si128((__m128i *)(dst + stride), packed);
    dst += 2 * stride;
    left += 2;
  }
}

void vpx_tm_predictor_32x32_avx2(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  __m256i top_left = _mm256_set1_epi16(above[-1]);
  __m256i zero = _mm256_setzero_si256();
  __m256i A = _mm256_load_si256((const __m256i *)above);
  // 0-7 || 16-23
  __m256i A0 = _mm256_sub_epi16(_mm256_unpacklo_epi8(A, zero), top_left);
  // 8-15 || 24-31
  __m256i A1 = _mm256_sub_epi16(_mm256_unpackhi_epi8(A, zero), top_left);

  for (int i = 0; i < 16; i++) {
    __m256i L0 = _mm256_set1_epi16(left[0]);
    __m256i D0 = _mm256_add_epi16(A0, L0);
    __m256i D1 = _mm256_add_epi16(A1, L0);

    // 256-bit packuswb packs within 128-bit lanes
    // bringing back 0-15 || 16-31
    __m256i packed = _mm256_packus_epi16(D0, D1);
    _mm256_store_si256((__m256i *)dst, packed);

    __m256i L1 = _mm256_set1_epi16(left[1]);
    D0 = _mm256_add_epi16(A0, L1);
    D1 = _mm256_add_epi16(A1, L1);

    packed = _mm256_packus_epi16(D0, D1);
    _mm256_store_si256((__m256i *)(dst + stride), packed);

    dst += 2 * stride;
    left += 2;
  }
}
