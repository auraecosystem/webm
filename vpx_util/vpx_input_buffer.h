/*
 *  Copyright (c) 2026 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_UTIL_VPX_INPUT_BUFFER_H_
#define VPX_VPX_UTIL_VPX_INPUT_BUFFER_H_

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"

#ifdef __cplusplus
extern "C" {
#endif

// A non-owning byte range with a nonwrapping end address.
typedef struct vpx_input_buffer {
  const uint8_t *data;
  size_t size;
} vpx_input_buffer;

// Helpers returning a success flag leave cursor and output arguments unchanged
// on error.

static INLINE int vpx_input_buffer_is_valid(const vpx_input_buffer *buffer) {
  return buffer != NULL &&
         ((buffer->data == NULL && buffer->size == 0) ||
          (buffer->data != NULL &&
           buffer->size <= UINTPTR_MAX - (uintptr_t)buffer->data));
}

static INLINE int vpx_input_buffer_init(vpx_input_buffer *buffer,
                                        const uint8_t *data, size_t size) {
  if (buffer == NULL || (data == NULL && size != 0) ||
      (data != NULL && size > UINTPTR_MAX - (uintptr_t)data)) {
    return 0;
  }
  buffer->data = data;
  buffer->size = size;
  return 1;
}

#if defined(__clang__)
#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#define VPX_INPUT_BUFFER_RESTORE_DIAGNOSTIC
#endif
#endif

// Keep raw pointer arithmetic and indexing in this block.
static INLINE int vpx_input_buffer_at(const vpx_input_buffer *buffer,
                                      size_t offset, const uint8_t **data) {
  if (data == NULL || !vpx_input_buffer_is_valid(buffer) ||
      offset > buffer->size) {
    return 0;
  }
  *data = offset == 0 ? buffer->data : buffer->data + offset;
  return 1;
}

static INLINE int vpx_input_buffer_read(const vpx_input_buffer *buffer,
                                        size_t offset, uint8_t *value) {
  if (value == NULL || !vpx_input_buffer_is_valid(buffer) ||
      offset >= buffer->size) {
    return 0;
  }
  *value = buffer->data[offset];
  return 1;
}

#if defined(VPX_INPUT_BUFFER_RESTORE_DIAGNOSTIC)
#pragma clang diagnostic pop
#undef VPX_INPUT_BUFFER_RESTORE_DIAGNOSTIC
#endif

static INLINE int vpx_input_buffer_skip(vpx_input_buffer *buffer, size_t size) {
  const uint8_t *data;
  if (!vpx_input_buffer_at(buffer, size, &data)) return 0;
  buffer->data = data;
  buffer->size -= size;
  return 1;
}

static INLINE int vpx_input_buffer_take(vpx_input_buffer *buffer, size_t size,
                                        vpx_input_buffer *result) {
  const uint8_t *data;
  if (result == NULL || result == buffer ||
      !vpx_input_buffer_is_valid(buffer) || size > buffer->size ||
      !vpx_input_buffer_at(buffer, size, &data)) {
    return 0;
  }
  result->data = buffer->data;
  result->size = size;
  buffer->data = data;
  buffer->size -= size;
  return 1;
}

static INLINE int vpx_input_buffer_subrange(const vpx_input_buffer *buffer,
                                            size_t offset, size_t size,
                                            vpx_input_buffer *result) {
  const uint8_t *data;
  if (result == NULL || !vpx_input_buffer_is_valid(buffer) ||
      offset > buffer->size || size > buffer->size - offset ||
      !vpx_input_buffer_at(buffer, offset, &data)) {
    return 0;
  }
  result->data = data;
  result->size = size;
  return 1;
}

static INLINE int vpx_input_buffer_copy(const vpx_input_buffer *buffer,
                                        size_t offset, void *output,
                                        size_t size) {
  vpx_input_buffer source;
  if (output == NULL && size != 0) return 0;
  if (!vpx_input_buffer_subrange(buffer, offset, size, &source)) return 0;
  if (size != 0) memcpy(output, source.data, size);
  return 1;
}

static INLINE int vpx_input_buffer_read_le24(const vpx_input_buffer *buffer,
                                             size_t offset, uint32_t *value) {
  uint8_t bytes[3];
  if (value == NULL ||
      !vpx_input_buffer_copy(buffer, offset, bytes, sizeof(bytes))) {
    return 0;
  }
  *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16);
  return 1;
}

static INLINE int vpx_input_buffer_read_be32(const vpx_input_buffer *buffer,
                                             size_t offset, uint32_t *value) {
  uint8_t bytes[4];
  if (value == NULL ||
      !vpx_input_buffer_copy(buffer, offset, bytes, sizeof(bytes))) {
    return 0;
  }
  *value = ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
  return 1;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_UTIL_VPX_INPUT_BUFFER_H_
