/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/ivf_video_source.h"
#include "vp8/decoder/dboolhuff.h"

namespace {
// In a real use the 'decrypt_state' parameter will be a pointer to a struct
// with whatever internal state the decryptor uses. For testing we'll just
// xor with a constant key, and decrypt_state will point to the start of
// the original buffer.
const uint8_t test_key[16] = { 0x01, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
                               0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0xf0 };

void encrypt_buffer(const uint8_t *src, uint8_t *dst, size_t size,
                    ptrdiff_t offset) {
  for (size_t i = 0; i < size; ++i) {
    dst[i] = src[i] ^ test_key[(offset + i) & 15];
  }
}

void test_decrypt_cb(void *decrypt_state, const uint8_t *input, uint8_t *output,
                     int count) {
  encrypt_buffer(input, output, count,
                 input - reinterpret_cast<uint8_t *>(decrypt_state));
}

// Keep the bool decoder state at its previous size.
struct BoolDecoderSizeBaseline {
  const unsigned char *user_buffer_end;
  const unsigned char *user_buffer;
  VP8_BD_VALUE value;
  int count;
  unsigned int range;
  vpx_decrypt_cb decrypt_cb;
  void *decrypt_state;
};

static_assert(sizeof(BOOL_DECODER) == sizeof(BoolDecoderSizeBaseline),
              "BOOL_DECODER size changed");
static_assert(alignof(BOOL_DECODER) == alignof(BoolDecoderSizeBaseline),
              "BOOL_DECODER alignment changed");

struct DecryptCallState {
  const uint8_t *data;
  size_t size;
  int calls;
  bool invalid_range;
};

void range_checked_decrypt_cb(void *decrypt_state, const uint8_t *input,
                              uint8_t *output, int count) {
  DecryptCallState *const state =
      static_cast<DecryptCallState *>(decrypt_state);
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

namespace libvpx_test {

TEST(VP8, TestBitReaderAcceptsEmptyRange) {
  BOOL_DECODER br;
  DecryptCallState state = { nullptr, 0, 0, false };
  EXPECT_EQ(
      vp8dx_start_decode(&br, nullptr, 0, range_checked_decrypt_cb, &state), 0);
  EXPECT_EQ(br.input.data, nullptr);
  EXPECT_EQ(br.input.size, 0u);
  EXPECT_EQ(state.calls, 0);
}

TEST(VP8, TestBitReaderRejectsNullNonemptyRange) {
  BOOL_DECODER br;
  DecryptCallState state = { nullptr, 0, 0, false };
  EXPECT_EQ(
      vp8dx_start_decode(&br, nullptr, 1, range_checked_decrypt_cb, &state), 1);
  EXPECT_EQ(state.calls, 0);
}

TEST(VP8, TestBitReaderDecryptWindowsStayInRange) {
  uint8_t data[sizeof(VP8_BD_VALUE) + 2] = { 0 };
  for (size_t size = 1; size <= sizeof(data); ++size) {
    BOOL_DECODER plain;
    BOOL_DECODER decrypted;
    DecryptCallState state = { data, size, 0, false };

    ASSERT_EQ(vp8dx_start_decode(&plain, data, static_cast<unsigned int>(size),
                                 nullptr, nullptr),
              0);
    ASSERT_EQ(
        vp8dx_start_decode(&decrypted, data, static_cast<unsigned int>(size),
                           range_checked_decrypt_cb, &state),
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

TEST(TestDecrypt, DecryptWorksVp8) {
  libvpx_test::IVFVideoSource video("vp80-00-comprehensive-001.ivf");
  video.Init();

  vpx_codec_dec_cfg_t dec_cfg = vpx_codec_dec_cfg_t();
  VP8Decoder decoder(dec_cfg, 0);

  video.Begin();

  // no decryption
  vpx_codec_err_t res = decoder.DecodeFrame(video.cxdata(), video.frame_size());
  ASSERT_EQ(VPX_CODEC_OK, res) << decoder.DecodeError();

  // decrypt frame
  video.Next();

  std::vector<uint8_t> encrypted(video.frame_size());
  encrypt_buffer(video.cxdata(), &encrypted[0], video.frame_size(), 0);
  vpx_decrypt_init di = { test_decrypt_cb, &encrypted[0] };
  decoder.Control(VPXD_SET_DECRYPTOR, &di);

  res = decoder.DecodeFrame(&encrypted[0], encrypted.size());
  ASSERT_EQ(VPX_CODEC_OK, res) << decoder.DecodeError();
}

}  // namespace libvpx_test
