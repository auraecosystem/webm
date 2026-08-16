/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_BITREADER_BUFFER_H_
#define VPX_VPX_DSP_BITREADER_BUFFER_H_

#include <limits.h>

#include "vpx/vpx_integer.h"
#include "vpx_util/vpx_input_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*vpx_rb_error_handler)(void *data);

struct vpx_read_bit_buffer {
  vpx_input_buffer input;
  size_t bit_offset;

  void *error_handler_data;
  vpx_rb_error_handler error_handler;
};

int vpx_rb_init(struct vpx_read_bit_buffer *rb, const uint8_t *data,
                size_t size, vpx_rb_error_handler error_handler,
                void *error_handler_data);

int vpx_rb_skip_bits(struct vpx_read_bit_buffer *rb, size_t bits);

size_t vpx_rb_bytes_read(struct vpx_read_bit_buffer *rb);

int vpx_rb_read_bit(struct vpx_read_bit_buffer *rb);

int vpx_rb_read_literal(struct vpx_read_bit_buffer *rb, int bits);

int vpx_rb_read_signed_literal(struct vpx_read_bit_buffer *rb, int bits);

int vpx_rb_read_inv_signed_literal(struct vpx_read_bit_buffer *rb, int bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_BITREADER_BUFFER_H_
