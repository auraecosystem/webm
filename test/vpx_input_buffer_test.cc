/*
 *  Copyright (c) 2026 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stddef.h>

#include "gtest/gtest.h"

#include "vpx_util/vpx_input_buffer.h"

TEST(InputBufferTest, ValidatesNullInput) {
  const uint8_t byte = 0;
  vpx_input_buffer input = { &byte, 1 };

  EXPECT_TRUE(vpx_input_buffer_init(&input, nullptr, 0));
  EXPECT_EQ(input.data, nullptr);
  EXPECT_EQ(input.size, 0u);

  input.data = &byte;
  input.size = 1;
  EXPECT_FALSE(vpx_input_buffer_init(&input, nullptr, 1));
  EXPECT_EQ(input.data, &byte);
  EXPECT_EQ(input.size, 1u);
}

TEST(InputBufferTest, RejectsInvalidCursorState) {
  vpx_input_buffer input = { nullptr, 1 };
  vpx_input_buffer result = { nullptr, 0 };
  EXPECT_FALSE(vpx_input_buffer_skip(&input, 0));
  EXPECT_FALSE(vpx_input_buffer_take(&input, 0, &result));
  EXPECT_FALSE(vpx_input_buffer_subrange(&input, 0, 0, &result));
  EXPECT_EQ(input.data, nullptr);
  EXPECT_EQ(input.size, 1u);
}

TEST(InputBufferTest, RejectsAddressWrapWithoutMutation) {
  const uint8_t byte = 0;
  // The synthetic pointer is never dereferenced. It exercises the integer
  // arithmetic that rejects a range whose end address would wrap.
  const uint8_t *const near_end =
      reinterpret_cast<const uint8_t *>(UINTPTR_MAX - 1);
  vpx_input_buffer input = { &byte, 1 };

  if (reinterpret_cast<uintptr_t>(near_end) != UINTPTR_MAX - 1) {
    GTEST_SKIP() << "Integer-to-pointer conversion did not preserve the value";
  }

  EXPECT_FALSE(vpx_input_buffer_init(&input, near_end, 2));
  EXPECT_EQ(input.data, &byte);
  EXPECT_EQ(input.size, 1u);

  input.data = near_end;
  input.size = 2;
  EXPECT_FALSE(vpx_input_buffer_is_valid(&input));
  EXPECT_FALSE(vpx_input_buffer_skip(&input, 1));
  EXPECT_EQ(input.data, near_end);
  EXPECT_EQ(input.size, 2u);
}

TEST(InputBufferTest, RejectsNullOutputs) {
  const uint8_t data[] = { 1, 2, 3, 4 };
  vpx_input_buffer input;
  ASSERT_TRUE(vpx_input_buffer_init(&input, data, sizeof(data)));

  EXPECT_FALSE(vpx_input_buffer_init(nullptr, data, sizeof(data)));
  EXPECT_FALSE(vpx_input_buffer_at(&input, 0, nullptr));
  EXPECT_FALSE(vpx_input_buffer_read(&input, 0, nullptr));
  EXPECT_FALSE(vpx_input_buffer_take(&input, 1, nullptr));
  EXPECT_FALSE(vpx_input_buffer_subrange(&input, 0, 1, nullptr));
  EXPECT_FALSE(vpx_input_buffer_copy(&input, 0, nullptr, 1));
  EXPECT_FALSE(vpx_input_buffer_read_le24(&input, 0, nullptr));
  EXPECT_FALSE(vpx_input_buffer_read_be32(&input, 0, nullptr));

  EXPECT_EQ(input.data, data);
  EXPECT_EQ(input.size, sizeof(data));
}

TEST(InputBufferTest, TakesAndSkipsCheckedRanges) {
  const uint8_t data[] = { 1, 2, 3, 4 };
  vpx_input_buffer input;
  vpx_input_buffer prefix;
  ASSERT_TRUE(vpx_input_buffer_init(&input, data, sizeof(data)));

  ASSERT_TRUE(vpx_input_buffer_take(&input, 2, &prefix));
  EXPECT_EQ(prefix.data, data);
  EXPECT_EQ(prefix.size, 2u);
  EXPECT_EQ(input.data, data + 2);
  EXPECT_EQ(input.size, 2u);

  ASSERT_TRUE(vpx_input_buffer_skip(&input, 2));
  EXPECT_EQ(input.data, data + sizeof(data));
  EXPECT_EQ(input.size, 0u);

  EXPECT_FALSE(vpx_input_buffer_skip(&input, 1));
  EXPECT_EQ(input.data, data + sizeof(data));
  EXPECT_EQ(input.size, 0u);
}

TEST(InputBufferTest, RejectsInvalidSubrangesWithoutMutation) {
  const uint8_t data[] = { 1, 2, 3, 4 };
  vpx_input_buffer input;
  vpx_input_buffer result = { nullptr, 0 };
  ASSERT_TRUE(vpx_input_buffer_init(&input, data, sizeof(data)));

  ASSERT_TRUE(vpx_input_buffer_subrange(&input, 1, 2, &result));
  EXPECT_EQ(result.data, data + 1);
  EXPECT_EQ(result.size, 2u);

  EXPECT_FALSE(vpx_input_buffer_subrange(&input, sizeof(data) + 1, 0, &result));
  EXPECT_EQ(result.data, data + 1);
  EXPECT_EQ(result.size, 2u);

  EXPECT_FALSE(vpx_input_buffer_take(&input, sizeof(data) + 1, &result));
  EXPECT_EQ(input.data, data);
  EXPECT_EQ(input.size, sizeof(data));
  EXPECT_EQ(result.data, data + 1);
  EXPECT_EQ(result.size, 2u);
}

TEST(InputBufferTest, RejectsAliasedTakeWithoutMutation) {
  const uint8_t data[] = { 1, 2, 3, 4 };
  vpx_input_buffer input;
  ASSERT_TRUE(vpx_input_buffer_init(&input, data, sizeof(data)));

  EXPECT_FALSE(vpx_input_buffer_take(&input, 2, &input));
  EXPECT_EQ(input.data, data);
  EXPECT_EQ(input.size, sizeof(data));
}

TEST(InputBufferTest, ReadsWithinBounds) {
  const uint8_t data[] = { 0x12, 0x34 };
  uint8_t value = 0;
  vpx_input_buffer input;
  ASSERT_TRUE(vpx_input_buffer_init(&input, data, sizeof(data)));

  EXPECT_TRUE(vpx_input_buffer_read(&input, 1, &value));
  EXPECT_EQ(value, 0x34);
  EXPECT_FALSE(vpx_input_buffer_read(&input, sizeof(data), &value));
  EXPECT_EQ(value, 0x34);
}

TEST(InputBufferTest, ReadsEndianValues) {
  const uint8_t data[] = { 0x01, 0x23, 0x45, 0x67 };
  uint32_t value = 0;
  vpx_input_buffer input;
  ASSERT_TRUE(vpx_input_buffer_init(&input, data, sizeof(data)));

  EXPECT_TRUE(vpx_input_buffer_read_le24(&input, 0, &value));
  EXPECT_EQ(value, 0x00452301u);
  EXPECT_TRUE(vpx_input_buffer_read_be32(&input, 0, &value));
  EXPECT_EQ(value, 0x01234567u);

  value = 9;
  EXPECT_FALSE(vpx_input_buffer_read_le24(&input, 2, &value));
  EXPECT_EQ(value, 9u);
  EXPECT_FALSE(vpx_input_buffer_read_be32(&input, 1, &value));
  EXPECT_EQ(value, 9u);
}

TEST(InputBufferTest, RejectsOverflowingOffsets) {
  const uint8_t byte = 0;
  vpx_input_buffer input;
  vpx_input_buffer result = { nullptr, 0 };
  ASSERT_TRUE(vpx_input_buffer_init(&input, &byte, 1));

  EXPECT_FALSE(vpx_input_buffer_subrange(&input, SIZE_MAX, 1, &result));
  EXPECT_EQ(input.data, &byte);
  EXPECT_EQ(input.size, 1u);
}
