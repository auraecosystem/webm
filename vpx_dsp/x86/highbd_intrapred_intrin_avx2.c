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

static INLINE __m256i avg3_epu16_avx2(const __m256i *x, const __m256i *y,
                                      const __m256i *z) {
  const __m256i one = _mm256_set1_epi16(1);
  const __m256i a = _mm256_avg_epu16(*x, *z);
  const __m256i b =
      _mm256_subs_epu16(a, _mm256_and_si256(_mm256_xor_si256(*x, *z), one));
  return _mm256_avg_epu16(b, *y);
}

/*
 palignr in AVX2 operates in-lane
 hi: hi_hi | hi_lo
 mid hi_lo | lo_hi
 lo: lo_hi | lo_lo
*/
#define ALIGNR_256(res, hi, lo, i)                       \
  do {                                                   \
    __m256i _mid = _mm256_permute2x128_si256(hi, lo, 3); \
    res = _mm256_alignr_epi8(_mid, lo, (i));             \
  } while (0)

void vpx_highbd_dc_predictor_16x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                        const uint16_t *above,
                                        const uint16_t *left, int bd) {
  (void)bd;
  __m256i sum =
      _mm256_add_epi16(*(const __m256i *)above, *(const __m256i *)left);

  __m128i sum128 = _mm_add_epi16(_mm256_castsi256_si128(sum),
                                 _mm256_extracti128_si256(sum, 1));
  sum128 = _mm_cvtepu16_epi32(
      _mm_add_epi16(sum128, _mm_unpackhi_epi64(sum128, sum128)));
  sum128 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
  sum128 =
      _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 1, 1, 1)));
  sum128 = _mm_add_epi32(sum128, _mm_cvtsi32_si128(16));
  sum128 = _mm_srli_epi32(sum128, 5);

  __m256i v = _mm256_broadcastw_epi16(sum128);

  for (int i = 0; i < 16; ++i) {
    _mm256_store_si256((__m256i *)dst, v);
    dst += stride;
  }
}

static INLINE void dc_store_sum_16(uint16_t *dst, ptrdiff_t stride,
                                   const __m256i *sum) {
  __m128i sum128 = _mm_add_epi16(_mm256_castsi256_si128(*sum),
                                 _mm256_extracti128_si256(*sum, 1));
  sum128 = _mm_cvtepu16_epi32(
      _mm_add_epi16(sum128, _mm_unpackhi_epi64(sum128, sum128)));
  sum128 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
  sum128 =
      _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 1, 1, 1)));
  sum128 = _mm_add_epi16(sum128, _mm_srli_epi32(sum128, 16));
  sum128 = _mm_add_epi32(sum128, _mm_cvtsi32_si128(8));
  sum128 = _mm_srli_epi32(sum128, 4);

  __m256i v = _mm256_broadcastw_epi16(sum128);

  for (int i = 0; i < 16; ++i) {
    _mm256_store_si256((__m256i *)dst, v);
    dst += stride;
  }
}

void vpx_highbd_dc_left_predictor_16x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                             const uint16_t *above,
                                             const uint16_t *left, int bd) {
  (void)above;
  (void)bd;
  dc_store_sum_16(dst, stride, (const __m256i *)left);
}

void vpx_highbd_dc_top_predictor_16x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                            const uint16_t *above,
                                            const uint16_t *left, int bd) {
  (void)left;
  (void)bd;
  dc_store_sum_16(dst, stride, (const __m256i *)above);
}

void vpx_highbd_dc_128_predictor_16x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                            const uint16_t *above,
                                            const uint16_t *left, int bd) {
  (void)above;
  (void)left;
  const __m256i v = _mm256_set1_epi16(1 << (bd - 1));
  for (int i = 0; i < 16; ++i) {
    _mm256_store_si256((__m256i *)dst, v);
    dst += stride;
  }
}

#define DC_STORE_SUM_32(offset, shift)                                  \
  do {                                                                  \
    __m128i sum128 = _mm_add_epi16(_mm256_castsi256_si128(sum),         \
                                   _mm256_extracti128_si256(sum, 1));   \
    sum128 = _mm_cvtepu16_epi32(                                        \
        _mm_add_epi16(sum128, _mm_unpackhi_epi64(sum128, sum128)));     \
    sum128 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128)); \
    sum128 = _mm_add_epi32(                                             \
        sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 1, 1, 1)));    \
    sum128 = _mm_add_epi32(sum128, _mm_cvtsi32_si128(offset));          \
    sum128 = _mm_srli_epi32(sum128, shift);                             \
    __m256i v = _mm256_broadcastw_epi16(sum128);                        \
    for (int i = 0; i < 32; ++i) {                                      \
      _mm256_store_si256((__m256i *)dst, v);                            \
      _mm256_store_si256((__m256i *)(dst + 16), v);                     \
      dst += stride;                                                    \
    }                                                                   \
  } while (0)

void vpx_highbd_dc_predictor_32x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                        const uint16_t *above,
                                        const uint16_t *left, int bd) {
  (void)bd;
  __m256i sum0 =
      _mm256_add_epi16(*(const __m256i *)above, *(const __m256i *)left);
  __m256i sum1 = _mm256_add_epi16(*(const __m256i *)(above + 16),
                                  *(const __m256i *)(left + 16));
  __m256i sum = _mm256_add_epi16(sum0, sum1);
  DC_STORE_SUM_32(32, 6);
}

void vpx_highbd_dc_left_predictor_32x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                             const uint16_t *above,
                                             const uint16_t *left, int bd) {
  (void)above;
  (void)bd;
  __m256i sum =
      _mm256_add_epi16(*(const __m256i *)left, *(const __m256i *)(left + 16));
  DC_STORE_SUM_32(16, 5);
}

void vpx_highbd_dc_top_predictor_32x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                            const uint16_t *above,
                                            const uint16_t *left, int bd) {
  (void)left;
  (void)bd;
  __m256i sum =
      _mm256_add_epi16(*(const __m256i *)above, *(const __m256i *)(above + 16));
  DC_STORE_SUM_32(16, 5);
}

void vpx_highbd_dc_128_predictor_32x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                            const uint16_t *above,
                                            const uint16_t *left, int bd) {
  (void)above;
  (void)left;
  const __m256i v = _mm256_set1_epi16(1 << (bd - 1));
  for (int i = 0; i < 32; ++i) {
    _mm256_store_si256((__m256i *)dst, v);
    _mm256_store_si256((__m256i *)(dst + 16), v);
    dst += stride;
  }
}

void vpx_highbd_d63_predictor_16x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                         const uint16_t *above,
                                         const uint16_t *left, int bd) {
  const __m256i A = _mm256_load_si256((const __m256i *)above);
  const __m256i AR = _mm256_set1_epi16(above[15]);

  __m256i B;
  ALIGNR_256(B, AR, A, 2);
  __m256i C;
  ALIGNR_256(C, AR, A, 4);

  __m256i avg2 = _mm256_avg_epu16(A, B);
  __m256i avg3 = avg3_epu16_avx2(&A, &B, &C);

  (void)left;
  (void)bd;

  __m256i ar_avg2 = _mm256_permute2x128_si256(AR, avg2, 3);
  __m256i ar_avg3 = _mm256_permute2x128_si256(AR, avg3, 3);

  __m256i row0 = avg2;
  __m256i row1 = avg3;
  _mm256_store_si256((__m256i *)dst, row0);
  dst += stride;
  _mm256_store_si256((__m256i *)dst, row1);
  dst += stride;

#define D63_STORE_2x16(i)                          \
  do {                                             \
    row0 = _mm256_alignr_epi8(ar_avg2, avg2, (i)); \
    row1 = _mm256_alignr_epi8(ar_avg3, avg3, (i)); \
    _mm256_store_si256((__m256i *)dst, row0);      \
    dst += stride;                                 \
    _mm256_store_si256((__m256i *)dst, row1);      \
    dst += stride;                                 \
  } while (0)

  D63_STORE_2x16(2);
  D63_STORE_2x16(4);
  D63_STORE_2x16(6);
  D63_STORE_2x16(8);
  D63_STORE_2x16(10);
  D63_STORE_2x16(12);
  D63_STORE_2x16(14);
#undef D63_STORE_2x16
}

static INLINE void d63_store_16x32_avx2(
    uint16_t **dst, const ptrdiff_t stride, const __m256i *avg2_low,
    const __m256i *avg2_mid_low, const __m256i *avg2_high,
    const __m256i *avg2_ar_high, const __m256i *avg3_low,
    const __m256i *avg3_mid_low, const __m256i *avg3_high,
    const __m256i *avg3_ar_high) {
  __m256i row0_0 = *avg2_low;
  __m256i row0_1 = *avg2_high;
  __m256i row1_0 = *avg3_low;
  __m256i row1_1 = *avg3_high;

  _mm256_store_si256((__m256i *)*dst, row0_0);
  _mm256_store_si256((__m256i *)(*dst + 16), row0_1);
  *dst += stride;

  _mm256_store_si256((__m256i *)*dst, row1_0);
  _mm256_store_si256((__m256i *)(*dst + 16), row1_1);
  *dst += stride;

#define D63_STORE_2x32(i)                                        \
  do {                                                           \
    row0_0 = _mm256_alignr_epi8(*avg2_mid_low, *avg2_low, (i));  \
    row0_1 = _mm256_alignr_epi8(*avg2_ar_high, *avg2_high, (i)); \
    row1_0 = _mm256_alignr_epi8(*avg3_mid_low, *avg3_low, (i));  \
    row1_1 = _mm256_alignr_epi8(*avg3_ar_high, *avg3_high, (i)); \
    _mm256_store_si256((__m256i *)*dst, row0_0);                 \
    _mm256_store_si256((__m256i *)(*dst + 16), row0_1);          \
    *dst += stride;                                              \
    _mm256_store_si256((__m256i *)*dst, row1_0);                 \
    _mm256_store_si256((__m256i *)(*dst + 16), row1_1);          \
    *dst += stride;                                              \
  } while (0)

  D63_STORE_2x32(2);
  D63_STORE_2x32(4);
  D63_STORE_2x32(6);
  D63_STORE_2x32(8);
  D63_STORE_2x32(10);
  D63_STORE_2x32(12);
  D63_STORE_2x32(14);
#undef D63_STORE_2x32
}

void vpx_highbd_d63_predictor_32x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                         const uint16_t *above,
                                         const uint16_t *left, int bd) {
  (void)left;
  (void)bd;

  const __m256i A0 = _mm256_load_si256((const __m256i *)(above));  // 0..15
  const __m256i A1 =
      _mm256_load_si256((const __m256i *)(above + 16));  // 16..31
  const __m256i AR = _mm256_set1_epi16(above[31]);

  __m256i B0;
  ALIGNR_256(B0, A1, A0, 2);
  __m256i B1;
  ALIGNR_256(B1, AR, A1, 2);

  __m256i C0;
  ALIGNR_256(C0, A1, A0, 4);
  __m256i C1;
  ALIGNR_256(C1, AR, A1, 4);

  __m256i avg2_low = _mm256_avg_epu16(A0, B0);
  __m256i avg2_high = _mm256_avg_epu16(A1, B1);
  __m256i avg3_low = avg3_epu16_avx2(&A0, &B0, &C0);
  __m256i avg3_high = avg3_epu16_avx2(&A1, &B1, &C1);

  __m256i avg2_mid_low = _mm256_permute2x128_si256(avg2_high, avg2_low, 3);
  __m256i avg2_ar_high = _mm256_permute2x128_si256(AR, avg2_high, 3);
  __m256i avg3_mid_low = _mm256_permute2x128_si256(avg3_high, avg3_low, 3);
  __m256i avg3_ar_high = _mm256_permute2x128_si256(AR, avg3_high, 3);

  d63_store_16x32_avx2(&dst, stride, &avg2_low, &avg2_mid_low, &avg2_high,
                       &avg2_ar_high, &avg3_low, &avg3_mid_low, &avg3_high,
                       &avg3_ar_high);
  d63_store_16x32_avx2(&dst, stride, &avg2_mid_low, &avg2_high, &avg2_ar_high,
                       &AR, &avg3_mid_low, &avg3_high, &avg3_ar_high, &AR);
}

static INLINE void d207_store_8x16_avx2(uint16_t **dst, const ptrdiff_t stride,
                                        const __m256i *ab, const __m256i *cd) {
  _mm256_store_si256((__m256i *)*dst, *ab);
  *dst += stride;

  __m256i mid = _mm256_permute2x128_si256(*cd, *ab, 3);

#define D207_STORE_16(hi, lo, i)                     \
  do {                                               \
    __m256i shift = _mm256_alignr_epi8(hi, lo, (i)); \
    _mm256_store_si256((__m256i *)*dst, shift);      \
    *dst += stride;                                  \
  } while (0)

  D207_STORE_16(mid, *ab, 4);
  D207_STORE_16(mid, *ab, 8);
  D207_STORE_16(mid, *ab, 12);

  _mm256_store_si256((__m256i *)*dst, mid);
  *dst += stride;

  D207_STORE_16(*cd, mid, 4);
  D207_STORE_16(*cd, mid, 8);
  D207_STORE_16(*cd, mid, 12);

#undef D207_STORE_16
}

void vpx_highbd_d207_predictor_16x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                          const uint16_t *above,
                                          const uint16_t *left, int bd) {
  const __m256i L = _mm256_load_si256((const __m256i *)left);
  const __m256i LR =
      _mm256_permute4x64_epi64(_mm256_shufflehi_epi16(L, 0xff), 0xff);

  __m256i B;
  ALIGNR_256(B, LR, L, 2);
  __m256i C;
  ALIGNR_256(C, LR, L, 4);

  const __m256i avg2 = _mm256_avg_epu16(L, B);
  const __m256i avg3 = avg3_epu16_avx2(&L, &B, &C);

  const __m256i out_ac = _mm256_unpacklo_epi16(avg2, avg3);
  const __m256i out_bd = _mm256_unpackhi_epi16(avg2, avg3);

  __m256i out_ab = _mm256_permute2x128_si256(out_ac, out_bd, 0x20);
  __m256i out_cd = _mm256_permute2x128_si256(out_ac, out_bd, 0x31);

  (void)above;
  (void)bd;

  d207_store_8x16_avx2(&dst, stride, &out_ab, &out_cd);
  d207_store_8x16_avx2(&dst, stride, &out_cd, &LR);
}

static INLINE void d207_store_8x32_avx2(uint16_t **dst, const ptrdiff_t stride,
                                        const __m256i *ab, const __m256i *cd,
                                        const __m256i *ef) {
  _mm256_store_si256((__m256i *)*dst, *ab);
  _mm256_store_si256((__m256i *)(*dst + 16), *cd);
  *dst += stride;

  __m256i mid_0 = _mm256_permute2x128_si256(*cd, *ab, 3);
  __m256i mid_1 = _mm256_permute2x128_si256(*ef, *cd, 3);

#define D207_STORE_32(ab_hi, ab_lo, cd_hi, cd_lo, i)          \
  do {                                                        \
    __m256i ab_shift = _mm256_alignr_epi8(ab_hi, ab_lo, (i)); \
    __m256i cd_shift = _mm256_alignr_epi8(cd_hi, cd_lo, (i)); \
    _mm256_store_si256((__m256i *)*dst, ab_shift);            \
    _mm256_store_si256((__m256i *)(*dst + 16), cd_shift);     \
    *dst += stride;                                           \
  } while (0)

  D207_STORE_32(mid_0, *ab, mid_1, *cd, 4);
  D207_STORE_32(mid_0, *ab, mid_1, *cd, 8);
  D207_STORE_32(mid_0, *ab, mid_1, *cd, 12);

  _mm256_store_si256((__m256i *)*dst, mid_0);
  _mm256_store_si256((__m256i *)(*dst + 16), mid_1);
  *dst += stride;

  D207_STORE_32(*cd, mid_0, *ef, mid_1, 4);
  D207_STORE_32(*cd, mid_0, *ef, mid_1, 8);
  D207_STORE_32(*cd, mid_0, *ef, mid_1, 12);

#undef D207_STORE_32
}

void vpx_highbd_d207_predictor_32x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                          const uint16_t *above,
                                          const uint16_t *left, int bd) {
  (void)above;
  (void)bd;

  const __m256i A0 = _mm256_load_si256((const __m256i *)left);         // 0..15
  const __m256i A1 = _mm256_load_si256((const __m256i *)(left + 16));  // 16..31

  const __m256i AR =
      _mm256_permute4x64_epi64(_mm256_shufflehi_epi16(A1, 0xff), 0xff);

  __m256i B0;
  ALIGNR_256(B0, A1, A0, 2);
  __m256i B1;
  ALIGNR_256(B1, AR, A1, 2);
  __m256i C0;
  ALIGNR_256(C0, A1, A0, 4);
  __m256i C1;
  ALIGNR_256(C1, AR, A1, 4);

  __m256i avg2_0 = _mm256_avg_epu16(A0, B0);
  __m256i avg2_1 = _mm256_avg_epu16(A1, B1);
  __m256i avg3_0 = avg3_epu16_avx2(&A0, &B0, &C0);
  __m256i avg3_1 = avg3_epu16_avx2(&A1, &B1, &C1);

  __m256i out_ac = _mm256_unpacklo_epi16(avg2_0, avg3_0);
  __m256i out_bd = _mm256_unpackhi_epi16(avg2_0, avg3_0);
  __m256i out_eg = _mm256_unpacklo_epi16(avg2_1, avg3_1);
  __m256i out_fh = _mm256_unpackhi_epi16(avg2_1, avg3_1);

  __m256i out_ab = _mm256_permute2x128_si256(out_ac, out_bd, 0x20);
  __m256i out_cd = _mm256_permute2x128_si256(out_ac, out_bd, 0x31);
  __m256i out_ef = _mm256_permute2x128_si256(out_eg, out_fh, 0x20);
  __m256i out_gh = _mm256_permute2x128_si256(out_eg, out_fh, 0x31);

  d207_store_8x32_avx2(&dst, stride, &out_ab, &out_cd, &out_ef);
  d207_store_8x32_avx2(&dst, stride, &out_cd, &out_ef, &out_gh);
  d207_store_8x32_avx2(&dst, stride, &out_ef, &out_gh, &AR);
  d207_store_8x32_avx2(&dst, stride, &out_gh, &AR, &AR);
}

void vpx_highbd_tm_predictor_8x8_avx2(uint16_t *dst, ptrdiff_t stride,
                                      const uint16_t *above,
                                      const uint16_t *left, int bd) {
  __m128i top_left = _mm_set1_epi16(above[-1]);
  __m128i A = _mm_sub_epi16(*(const __m128i *)above, top_left);

  __m128i bd_max = _mm_set1_epi16((1 << bd) - 1);
  __m128i bd_min = _mm_setzero_si128();

  for (int i = 0; i < 4; i++) {
    __m128i L0 = _mm_set1_epi16(left[0]);
    __m128i D = _mm_add_epi16(A, L0);

    D = _mm_min_epi16(D, bd_max);
    D = _mm_max_epi16(D, bd_min);

    _mm_store_si128((__m128i *)dst, D);

    __m128i L1 = _mm_set1_epi16(left[1]);
    D = _mm_add_epi16(A, L1);

    D = _mm_min_epi16(D, bd_max);
    D = _mm_max_epi16(D, bd_min);

    _mm_store_si128((__m128i *)(dst + stride), D);

    dst += 2 * stride;
    left += 2;
  }
}

void vpx_highbd_v_predictor_16x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                       const uint16_t *above,
                                       const uint16_t *left, int bd) {
  (void)left;
  (void)bd;
  __m256i A = _mm256_load_si256((const __m256i *)above);
  for (int i = 0; i < 16; ++i) {
    _mm256_store_si256((__m256i *)dst, A);
    dst += stride;
  }
}

void vpx_highbd_v_predictor_32x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                       const uint16_t *above,
                                       const uint16_t *left, int bd) {
  (void)left;
  (void)bd;
  __m256i A0 = _mm256_load_si256((const __m256i *)above);
  __m256i A1 = _mm256_load_si256((const __m256i *)(above + 16));
  for (int i = 0; i < 32; ++i) {
    _mm256_store_si256((__m256i *)dst, A0);
    _mm256_store_si256((__m256i *)(dst + 16), A1);
    dst += stride;
  }
}

void vpx_highbd_h_predictor_16x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                       const uint16_t *above,
                                       const uint16_t *left, int bd) {
  (void)above;
  (void)bd;
  __m256i L256 = _mm256_load_si256((const __m256i *)left);

  __m128i L = _mm256_castsi256_si128(L256);
  for (int i = 0; i < 8; ++i) {
    _mm256_store_si256((__m256i *)dst, _mm256_broadcastw_epi16(L));
    dst += stride;
    L = _mm_srli_si128(L, 2);
  }
  L = _mm256_extracti128_si256(L256, 1);
  for (int i = 0; i < 8; ++i) {
    _mm256_store_si256((__m256i *)dst, _mm256_broadcastw_epi16(L));
    dst += stride;
    L = _mm_srli_si128(L, 2);
  }
}

void vpx_highbd_h_predictor_32x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                       const uint16_t *above,
                                       const uint16_t *left, int bd) {
  (void)above;
  (void)bd;
  __m256i L0 = _mm256_load_si256((const __m256i *)left);
  __m256i L1 = _mm256_load_si256((const __m256i *)(left + 16));

  __m128i L = _mm256_castsi256_si128(L0);
  for (int i = 0; i < 8; ++i) {
    __m256i row = _mm256_broadcastw_epi16(L);
    _mm256_store_si256((__m256i *)dst, row);
    _mm256_store_si256((__m256i *)(dst + 16), row);
    dst += stride;
    L = _mm_srli_si128(L, 2);
  }

  L = _mm256_extracti128_si256(L0, 1);
  for (int i = 0; i < 8; ++i) {
    __m256i row = _mm256_broadcastw_epi16(L);
    _mm256_store_si256((__m256i *)dst, row);
    _mm256_store_si256((__m256i *)(dst + 16), row);
    dst += stride;
    L = _mm_srli_si128(L, 2);
  }

  L = _mm256_castsi256_si128(L1);
  for (int i = 0; i < 8; ++i) {
    __m256i row = _mm256_broadcastw_epi16(L);
    _mm256_store_si256((__m256i *)dst, row);
    _mm256_store_si256((__m256i *)(dst + 16), row);
    dst += stride;
    L = _mm_srli_si128(L, 2);
  }

  L = _mm256_extracti128_si256(L1, 1);
  for (int i = 0; i < 8; ++i) {
    __m256i row = _mm256_broadcastw_epi16(L);
    _mm256_store_si256((__m256i *)dst, row);
    _mm256_store_si256((__m256i *)(dst + 16), row);
    dst += stride;
    L = _mm_srli_si128(L, 2);
  }
}

void vpx_highbd_tm_predictor_16x16_avx2(uint16_t *dst, ptrdiff_t stride,
                                        const uint16_t *above,
                                        const uint16_t *left, int bd) {
  __m256i top_left = _mm256_set1_epi16(above[-1]);
  __m256i A = _mm256_sub_epi16(*(const __m256i *)above, top_left);

  __m256i bd_max = _mm256_set1_epi16((1 << bd) - 1);
  __m256i bd_min = _mm256_setzero_si256();

  for (int i = 0; i < 8; i++) {
    __m256i L0 = _mm256_set1_epi16(left[0]);
    __m256i D = _mm256_add_epi16(A, L0);

    D = _mm256_min_epi16(D, bd_max);
    D = _mm256_max_epi16(D, bd_min);

    _mm256_store_si256((__m256i *)dst, D);

    __m256i L1 = _mm256_set1_epi16(left[1]);
    D = _mm256_add_epi16(A, L1);

    D = _mm256_min_epi16(D, bd_max);
    D = _mm256_max_epi16(D, bd_min);

    _mm256_store_si256((__m256i *)(dst + stride), D);

    dst += 2 * stride;
    left += 2;
  }
}

void vpx_highbd_tm_predictor_32x32_avx2(uint16_t *dst, ptrdiff_t stride,
                                        const uint16_t *above,
                                        const uint16_t *left, int bd) {
  __m256i top_left = _mm256_set1_epi16(above[-1]);
  __m256i A0 = _mm256_sub_epi16(*(const __m256i *)above, top_left);
  __m256i A1 = _mm256_sub_epi16(*(const __m256i *)(above + 16), top_left);

  __m256i bd_max = _mm256_set1_epi16((1 << bd) - 1);
  __m256i bd_min = _mm256_setzero_si256();

  for (int i = 0; i < 16; i++) {
    __m256i L0 = _mm256_set1_epi16(left[0]);
    __m256i D0 = _mm256_add_epi16(A0, L0);
    __m256i D1 = _mm256_add_epi16(A1, L0);

    D0 = _mm256_min_epi16(D0, bd_max);
    D0 = _mm256_max_epi16(D0, bd_min);
    D1 = _mm256_min_epi16(D1, bd_max);
    D1 = _mm256_max_epi16(D1, bd_min);

    _mm256_store_si256((__m256i *)dst, D0);
    _mm256_store_si256((__m256i *)(dst + 16), D1);

    __m256i L1 = _mm256_set1_epi16(left[1]);
    D0 = _mm256_add_epi16(A0, L1);
    D1 = _mm256_add_epi16(A1, L1);

    D0 = _mm256_min_epi16(D0, bd_max);
    D0 = _mm256_max_epi16(D0, bd_min);
    D1 = _mm256_min_epi16(D1, bd_max);
    D1 = _mm256_max_epi16(D1, bd_min);

    _mm256_store_si256((__m256i *)(dst + stride), D0);
    _mm256_store_si256((__m256i *)(dst + 16 + stride), D1);

    dst += 2 * stride;
    left += 2;
  }
}
