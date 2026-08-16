/*
 *  Copyright (c) 2026 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "gtest/gtest.h"

#include "vpx_dsp/bitreader.h"
#include "vpx_dsp/bitreader_buffer.h"

namespace {

// Keep the entropy reader state at its previous size.
struct VpxReaderSizeBaseline {
  BD_VALUE value;
  unsigned int range;
  int count;
  const uint8_t *buffer_end;
  const uint8_t *buffer;
  vpx_decrypt_cb decrypt_cb;
  void *decrypt_state;
  uint8_t clear_buffer[sizeof(BD_VALUE) + 1];
};

static_assert(sizeof(vpx_reader) == sizeof(VpxReaderSizeBaseline),
              "vpx_reader size changed");
static_assert(alignof(vpx_reader) == alignof(VpxReaderSizeBaseline),
              "vpx_reader alignment changed");

struct VpxReadBitBufferSizeBaseline {
  const uint8_t *bit_buffer;
  const uint8_t *bit_buffer_end;
  size_t bit_offset;
  void *error_handler_data;
  vpx_rb_error_handler error_handler;
};

static_assert(sizeof(vpx_read_bit_buffer) ==
                  sizeof(VpxReadBitBufferSizeBaseline),
              "vpx_read_bit_buffer size changed");
static_assert(alignof(vpx_read_bit_buffer) ==
                  alignof(VpxReadBitBufferSizeBaseline),
              "vpx_read_bit_buffer alignment changed");

void CountReadError(void *data) { ++*static_cast<int *>(data); }

struct DecryptCallState {
  const uint8_t *data;
  size_t size;
  int calls;
  bool invalid_range;
};

void RangeCheckedDecrypt(void *data, const uint8_t *input, uint8_t *output,
                         int count) {
  DecryptCallState *const state = static_cast<DecryptCallState *>(data);
  const uintptr_t base = reinterpret_cast<uintptr_t>(state->data);
  const uintptr_t address = reinterpret_cast<uintptr_t>(input);
  ++state->calls;
  if (count <= 0 || output == nullptr || address < base) {
    state->invalid_range = true;
    return;
  }
  const size_t offset = static_cast<size_t>(address - base);
  const size_t size = static_cast<size_t>(count);
  if (offset > state->size || size > state->size - offset) {
    state->invalid_range = true;
    return;
  }
  memcpy(output, input, size);
}

}  // namespace

TEST(VpxReaderTest, AcceptsEmptyRange) {
  vpx_reader reader;
  const vpx_input_buffer initial_input = { nullptr, 0 };
  size_t bytes_read = 1;
  DecryptCallState state = { nullptr, 0, 0, false };

  EXPECT_EQ(vpx_reader_init(&reader, nullptr, 0, RangeCheckedDecrypt, &state),
            0);
  EXPECT_TRUE(vpx_reader_bytes_read(&reader, &initial_input, &bytes_read));
  EXPECT_EQ(bytes_read, 0u);
  EXPECT_EQ(reader.input.data, nullptr);
  EXPECT_EQ(reader.input.size, 0u);
  EXPECT_EQ(state.calls, 0);
}

TEST(VpxReaderTest, RejectsNullNonemptyRange) {
  vpx_reader reader;
  DecryptCallState state = { nullptr, 0, 0, false };

  EXPECT_EQ(vpx_reader_init(&reader, nullptr, 1, RangeCheckedDecrypt, &state),
            1);
  EXPECT_EQ(state.calls, 0);
}

TEST(VpxReaderTest, DecryptWindowsStayInRange) {
  uint8_t data[sizeof(BD_VALUE) + 2] = { 0 };
  for (size_t size = 1; size <= sizeof(data); ++size) {
    vpx_reader plain;
    vpx_reader decrypted;
    DecryptCallState state = { data, size, 0, false };

    ASSERT_EQ(vpx_reader_init(&plain, data, size, nullptr, nullptr), 0);
    ASSERT_EQ(
        vpx_reader_init(&decrypted, data, size, RangeCheckedDecrypt, &state),
        0);
    EXPECT_GT(state.calls, 0);
    EXPECT_FALSE(state.invalid_range);
    EXPECT_EQ(decrypted.input.data, plain.input.data);
    EXPECT_EQ(decrypted.input.size, plain.input.size);
    EXPECT_EQ(decrypted.value, plain.value);
    EXPECT_EQ(decrypted.count, plain.count);
    EXPECT_EQ(decrypted.range, plain.range);
  }
}

TEST(VpxReaderTest, BytesReadMatchesLegacyRewind) {
  const uint8_t data[32] = { 0 };
  const int counts[] = { CHAR_BIT,         CHAR_BIT + 1,      2 * CHAR_BIT,
                         2 * CHAR_BIT + 1, BD_VALUE_SIZE - 1, BD_VALUE_SIZE };
  const size_t consumed = 12;
  const vpx_input_buffer initial_input = { data, sizeof(data) };
  size_t bytes_read;
  vpx_reader reader;

  ASSERT_EQ(vpx_reader_init(&reader, data, sizeof(data), nullptr, nullptr), 0);
  reader.input.data = data + consumed;
  reader.input.size = sizeof(data) - consumed;
  for (const int count : counts) {
    const size_t buffered = count > CHAR_BIT && count < BD_VALUE_SIZE
                                ? static_cast<size_t>(count - 1) / CHAR_BIT
                                : 0;
    reader.count = count;
    ASSERT_TRUE(vpx_reader_bytes_read(&reader, &initial_input, &bytes_read));
    EXPECT_EQ(bytes_read, consumed - buffered) << "count=" << count;
  }

  reader.input.data = data + 2;
  reader.input.size = sizeof(data) - 2;
  reader.count = BD_VALUE_SIZE - 1;
  ASSERT_TRUE(vpx_reader_bytes_read(&reader, &initial_input, &bytes_read));
  EXPECT_EQ(bytes_read, 0u);
}

TEST(VpxReaderTest, BytesReadRejectsUnrelatedCursor) {
  const uint8_t data[4] = { 0 };
  const uint8_t other[4] = { 0 };
  const vpx_input_buffer initial_input = { data, sizeof(data) };
  vpx_reader reader;
  size_t bytes_read = 9;

  ASSERT_EQ(vpx_reader_init(&reader, data, sizeof(data), nullptr, nullptr), 0);
  reader.input.data = other + 2;
  reader.input.size = 2;
  EXPECT_FALSE(vpx_reader_bytes_read(&reader, &initial_input, &bytes_read));
  EXPECT_EQ(bytes_read, 9u);
}

TEST(VpxReadBitBufferTest, RejectsHugeOffset) {
  const uint8_t byte = 0;
  int errors = 0;
  struct vpx_read_bit_buffer reader;
  ASSERT_TRUE(vpx_rb_init(&reader, &byte, 1, CountReadError, &errors));
  reader.bit_offset = SIZE_MAX;

  EXPECT_EQ(vpx_rb_read_bit(&reader), 0);
  EXPECT_EQ(errors, 1);
  EXPECT_EQ(vpx_rb_bytes_read(&reader), SIZE_MAX / CHAR_BIT + 1);
}

TEST(VpxReadBitBufferTest, RejectsInvalidRange) {
  int errors = 0;
  struct vpx_read_bit_buffer reader;
  reader.input.data = nullptr;
  reader.input.size = 1;
  reader.bit_offset = 0;
  reader.error_handler = CountReadError;
  reader.error_handler_data = &errors;

  EXPECT_FALSE(vpx_rb_skip_bits(&reader, 0));
  EXPECT_EQ(reader.bit_offset, 0u);
  EXPECT_EQ(errors, 1);
}

TEST(VpxReadBitBufferTest, SkipsWithinRange) {
  const uint8_t byte = 0;
  int errors = 0;
  struct vpx_read_bit_buffer reader;
  ASSERT_TRUE(vpx_rb_init(&reader, &byte, 1, CountReadError, &errors));

  EXPECT_TRUE(vpx_rb_skip_bits(&reader, 7));
  EXPECT_TRUE(vpx_rb_skip_bits(&reader, 1));
  EXPECT_EQ(vpx_rb_bytes_read(&reader), 1u);
  EXPECT_FALSE(vpx_rb_skip_bits(&reader, 1));
  EXPECT_EQ(reader.bit_offset, 8u);
  EXPECT_EQ(errors, 1);

  reader.bit_offset = 1;
  EXPECT_FALSE(vpx_rb_skip_bits(&reader, SIZE_MAX));
  EXPECT_EQ(reader.bit_offset, 1u);
  EXPECT_EQ(errors, 2);
}
