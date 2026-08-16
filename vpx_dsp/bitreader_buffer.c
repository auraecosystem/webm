/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "./vpx_config.h"
#include "./bitreader_buffer.h"

int vpx_rb_init(struct vpx_read_bit_buffer *rb, const uint8_t *data,
                size_t size, vpx_rb_error_handler error_handler,
                void *error_handler_data) {
  if (!vpx_input_buffer_init(&rb->input, data, size)) return 0;
  rb->bit_offset = 0;
  rb->error_handler = error_handler;
  rb->error_handler_data = error_handler_data;
  return 1;
}

int vpx_rb_skip_bits(struct vpx_read_bit_buffer *rb, size_t bits) {
  size_t new_offset;
  size_t bytes;
  size_t remainder;
  if (!vpx_input_buffer_is_valid(&rb->input)) goto fail;
  if (bits > SIZE_MAX - rb->bit_offset) goto fail;
  new_offset = rb->bit_offset + bits;
  bytes = new_offset / CHAR_BIT;
  remainder = new_offset % CHAR_BIT;
  if (bytes > rb->input.size || (bytes == rb->input.size && remainder != 0)) {
    goto fail;
  }
  rb->bit_offset = new_offset;
  return 1;

fail:
  if (rb->error_handler != NULL) rb->error_handler(rb->error_handler_data);
  return 0;
}

size_t vpx_rb_bytes_read(struct vpx_read_bit_buffer *rb) {
  return rb->bit_offset / CHAR_BIT + (rb->bit_offset % CHAR_BIT != 0);
}

int vpx_rb_read_bit(struct vpx_read_bit_buffer *rb) {
  const size_t off = rb->bit_offset;
  const size_t p = off >> 3;
  const int q = 7 - (int)(off & 0x7);
  uint8_t byte;
  if (off != SIZE_MAX && vpx_input_buffer_read(&rb->input, p, &byte)) {
    const int bit = (byte >> q) & 1;
    rb->bit_offset = off + 1;
    return bit;
  } else {
    if (rb->error_handler != NULL) rb->error_handler(rb->error_handler_data);
    return 0;
  }
}

int vpx_rb_read_literal(struct vpx_read_bit_buffer *rb, int bits) {
  int value = 0, bit;
  for (bit = bits - 1; bit >= 0; bit--) value |= vpx_rb_read_bit(rb) << bit;
  return value;
}

int vpx_rb_read_signed_literal(struct vpx_read_bit_buffer *rb, int bits) {
  const int value = vpx_rb_read_literal(rb, bits);
  return vpx_rb_read_bit(rb) ? -value : value;
}

int vpx_rb_read_inv_signed_literal(struct vpx_read_bit_buffer *rb, int bits) {
  return vpx_rb_read_signed_literal(rb, bits);
}
