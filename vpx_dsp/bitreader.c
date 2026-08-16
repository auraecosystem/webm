/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <assert.h>
#include <stdlib.h>

#include "./vpx_config.h"

#include "vpx_dsp/bitreader.h"
#include "vpx_dsp/prob.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_ports/mem.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_util/endian_inl.h"

int vpx_reader_init(vpx_reader *r, const uint8_t *buffer, size_t size,
                    vpx_decrypt_cb decrypt_cb, void *decrypt_state) {
  if (!vpx_input_buffer_init(&r->input, buffer, size)) {
    return 1;
  } else {
    r->value = 0;
    r->count = -8;
    r->range = 255;
    r->decrypt_cb = decrypt_cb;
    r->decrypt_state = decrypt_state;
    vpx_reader_fill(r);
    return vpx_read_bit(r) != 0;  // marker bit
  }
}

void vpx_reader_fill(vpx_reader *r) {
  vpx_input_buffer input = r->input;
  const vpx_decrypt_cb decrypt_cb = r->decrypt_cb;
  void *const decrypt_state = r->decrypt_state;
  const uint8_t *buffer = input.data;
  vpx_input_buffer window;
  BD_VALUE value = r->value;
  int count = r->count;
  const size_t bytes_left = input.size;
  size_t window_size = bytes_left;
  size_t consumed = 0;
  int shift = BD_VALUE_SIZE - CHAR_BIT - (count + CHAR_BIT);

  if (decrypt_cb && bytes_left != 0) {
    const size_t n = VPXMIN(sizeof(r->clear_buffer), bytes_left);
    window_size = n;
    decrypt_cb(decrypt_state, buffer, r->clear_buffer, (int)n);
    buffer = r->clear_buffer;
  }
  window.data = buffer;
  window.size = window_size;
  if (bytes_left > sizeof(BD_VALUE)) {
    const int bits = (shift & 0xfffffff8) + CHAR_BIT;
    BD_VALUE nv;
    BD_VALUE big_endian_values;
    memcpy(&big_endian_values, buffer, sizeof(BD_VALUE));
#if SIZE_MAX == 0xffffffffffffffffULL
    big_endian_values = HToBE64(big_endian_values);
#else
    big_endian_values = HToBE32(big_endian_values);
#endif
    nv = big_endian_values >> (BD_VALUE_SIZE - bits);
    count += bits;
    consumed = (size_t)(bits >> 3);
    value = r->value | (nv << (shift & 0x7));
  } else {
    const size_t bits_left = bytes_left * CHAR_BIT;
    const int bits_over = (int)(shift + CHAR_BIT - (int)bits_left);
    int loop_end = 0;
    if (bits_over >= 0) {
      count += LOTS_OF_BITS;
      loop_end = bits_over;
    }

    if (bits_over < 0 || bits_left) {
      assert(vpx_input_buffer_is_valid(&window));
      assert(shift >= loop_end);
      assert(shift < BD_VALUE_SIZE);
      assert((size_t)(shift - loop_end) / CHAR_BIT < window.size);
#if defined(__clang__)
#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif
#endif
      while (shift >= loop_end) {
        count += CHAR_BIT;
        value |= (BD_VALUE)window.data[consumed] << shift;
        ++consumed;
        shift -= CHAR_BIT;
      }
#if defined(__clang__)
#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang diagnostic pop
#endif
#endif
    }
  }

  if (!vpx_input_buffer_skip(&input, consumed)) {
    input.size = 0;
    count += LOTS_OF_BITS;
  }
  r->input = input;
  r->value = value;
  r->count = count;
}

int vpx_reader_bytes_read(const vpx_reader *r,
                          const vpx_input_buffer *initial_input,
                          size_t *bytes_read) {
  const uint8_t *cursor;
  size_t buffered_bytes = 0;
  size_t consumed;

  if (bytes_read == NULL || !vpx_input_buffer_is_valid(initial_input) ||
      !vpx_input_buffer_is_valid(&r->input) ||
      r->input.size > initial_input->size) {
    return 0;
  }
  consumed = initial_input->size - r->input.size;
  if (!vpx_input_buffer_at(initial_input, consumed, &cursor) ||
      cursor != r->input.data) {
    return 0;
  }
  if (r->count > CHAR_BIT && r->count < BD_VALUE_SIZE) {
    buffered_bytes = (size_t)(r->count - 1) / CHAR_BIT;
  }
  *bytes_read = buffered_bytes < consumed ? consumed - buffered_bytes : 0;
  return 1;
}
