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

#include "dboolhuff.h"
#include "vp8/common/common.h"
#include "vpx_dsp/vpx_dsp_common.h"

int vp8dx_start_decode(BOOL_DECODER *br, const unsigned char *source,
                       unsigned int source_sz, vpx_decrypt_cb decrypt_cb,
                       void *decrypt_state) {
  if (!vpx_input_buffer_init(&br->input, source, source_sz)) return 1;

  br->value = 0;
  br->count = -8;
  br->range = 255;
  br->decrypt_cb = decrypt_cb;
  br->decrypt_state = decrypt_state;

  /* Populate the buffer */
  vp8dx_bool_decoder_fill(br);

  return 0;
}

void vp8dx_bool_decoder_fill(BOOL_DECODER *br) {
  vpx_input_buffer input = br->input;
  const vpx_decrypt_cb decrypt_cb = br->decrypt_cb;
  void *const decrypt_state = br->decrypt_state;
  const unsigned char *bufptr = input.data;
  vpx_input_buffer window_input;
  VP8_BD_VALUE value = br->value;
  int count = br->count;
  int shift = VP8_BD_VALUE_SIZE - CHAR_BIT - (count + CHAR_BIT);
  const size_t bytes_left = input.size;
  const size_t window = VPXMIN(sizeof(VP8_BD_VALUE) + 1, bytes_left);
  const size_t bits_left = window * CHAR_BIT;
  int x = shift + CHAR_BIT - (int)bits_left;
  int loop_end = 0;
  size_t consumed = 0;
  unsigned char decrypted[sizeof(VP8_BD_VALUE) + 1];

  if (decrypt_cb && window != 0) {
    const size_t n = window;
    decrypt_cb(decrypt_state, bufptr, decrypted, (int)n);
    bufptr = decrypted;
  }
  window_input.data = bufptr;
  window_input.size = decrypt_cb ? window : bytes_left;

  if (x >= 0) {
    count += VP8_LOTS_OF_BITS;
    loop_end = x;
  }

  if (x < 0 || bits_left) {
    assert(vpx_input_buffer_is_valid(&window_input));
    assert(shift >= loop_end);
    assert(shift < VP8_BD_VALUE_SIZE);
    assert((size_t)(shift - loop_end) / CHAR_BIT < window_input.size);
#if defined(__clang__)
#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif
#endif
    while (shift >= loop_end) {
      count += CHAR_BIT;
      value |= (VP8_BD_VALUE)window_input.data[consumed] << shift;
      ++consumed;
      shift -= CHAR_BIT;
    }
#if defined(__clang__)
#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang diagnostic pop
#endif
#endif
  }

  if (!vpx_input_buffer_skip(&input, consumed)) {
    input.size = 0;
    count += VP8_LOTS_OF_BITS;
  }
  br->input = input;
  br->value = value;
  br->count = count;
}
