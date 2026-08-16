/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/ivf_video_source.h"
#include "test/md5_helper.h"
#include "vp9/decoder/vp9_decoder.h"

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

struct RangeCheckedDecryptState {
  const uint8_t *data;
  size_t size;
  int calls;
  bool invalid_range;
};

void range_checked_decrypt_cb(void *decrypt_state, const uint8_t *input,
                              uint8_t *output, int count) {
  RangeCheckedDecryptState *const state =
      static_cast<RangeCheckedDecryptState *>(decrypt_state);
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
  encrypt_buffer(input, output, size, static_cast<ptrdiff_t>(offset));
}

std::string decoded_frame_md5(libvpx_test::Decoder *decoder) {
  libvpx_test::DxDataIterator frames = decoder->GetDxData();
  const vpx_image_t *const image = frames.Next();
  if (image == nullptr) {
    ADD_FAILURE() << "Decoder returned no frame";
    return std::string();
  }
  libvpx_test::MD5 md5;
  md5.Add(image);
  EXPECT_EQ(nullptr, frames.Next());
  return md5.Get();
}

}  // namespace

namespace libvpx_test {

TEST(VP9SuperframeIndex, EmptyInputDoesNotCallDecryptor) {
  uint32_t sizes[8] = { 0 };
  int frame_count = -1;
  RangeCheckedDecryptState state = { nullptr, 0, 0, false };

  EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
            vp9_parse_superframe_index(nullptr, 0, sizes, &frame_count,
                                       range_checked_decrypt_cb, &state));
  EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
            vp9_parse_superframe_index(nullptr, 1, sizes, &frame_count,
                                       range_checked_decrypt_cb, &state));
  EXPECT_EQ(state.calls, 0);
}

TEST(VP9SuperframeIndex, ParsesValidIndex) {
  const uint8_t data[] = { 0xaa, 0xbb, 0xc1, 1, 1, 0xc1 };
  uint32_t sizes[8] = { 0 };
  int frame_count = -1;

  EXPECT_EQ(VPX_CODEC_OK,
            vp9_parse_superframe_index(data, sizeof(data), sizes, &frame_count,
                                       nullptr, nullptr));
  EXPECT_EQ(frame_count, 2);
  EXPECT_EQ(sizes[0], 1u);
  EXPECT_EQ(sizes[1], 1u);

  std::vector<uint8_t> encrypted(data, data + sizeof(data));
  encrypt_buffer(data, encrypted.data(), encrypted.size(), 0);
  RangeCheckedDecryptState state = { encrypted.data(), encrypted.size(), 0,
                                     false };
  frame_count = -1;
  EXPECT_EQ(VPX_CODEC_OK, vp9_parse_superframe_index(
                              encrypted.data(), encrypted.size(), sizes,
                              &frame_count, range_checked_decrypt_cb, &state));
  EXPECT_EQ(frame_count, 2);
  EXPECT_EQ(sizes[0], 1u);
  EXPECT_EQ(sizes[1], 1u);
  EXPECT_GT(state.calls, 0);
  EXPECT_FALSE(state.invalid_range);
}

TEST(VP9SuperframeIndex, RejectsTruncatedIndex) {
  const uint8_t data[] = { 0xc0 };
  uint32_t sizes[8] = { 0 };
  int frame_count = -1;

  EXPECT_EQ(VPX_CODEC_CORRUPT_FRAME,
            vp9_parse_superframe_index(data, sizeof(data), sizes, &frame_count,
                                       nullptr, nullptr));
}

TEST(TestDecrypt, DecryptWorksVp9) {
  libvpx_test::IVFVideoSource video("vp90-2-05-resize.ivf");
  video.Init();

  vpx_codec_dec_cfg_t dec_cfg = vpx_codec_dec_cfg_t();
  VP9Decoder decoder(dec_cfg, 0);

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

TEST(TestDecrypt, ConcatenatedFramesWithoutIndex) {
  IVFVideoSource video("vp90-2-05-resize.ivf");
  video.Init();
  video.Begin();
  ASSERT_NE(video.cxdata(), nullptr);
  const std::vector<uint8_t> first_frame(video.cxdata(),
                                         video.cxdata() + video.frame_size());
  video.Next();
  ASSERT_NE(video.cxdata(), nullptr);
  const std::vector<uint8_t> second_frame(video.cxdata(),
                                          video.cxdata() + video.frame_size());

  vpx_codec_dec_cfg_t dec_cfg = vpx_codec_dec_cfg_t();
  VP9Decoder reference_decoder(dec_cfg, 0);
  ASSERT_EQ(VPX_CODEC_OK, reference_decoder.DecodeFrame(first_frame.data(),
                                                        first_frame.size()))
      << reference_decoder.DecodeError();
  ASSERT_EQ(VPX_CODEC_OK, reference_decoder.DecodeFrame(second_frame.data(),
                                                        second_frame.size()))
      << reference_decoder.DecodeError();
  const std::string expected_md5 = decoded_frame_md5(&reference_decoder);
  ASSERT_FALSE(expected_md5.empty());

  std::vector<uint8_t> packet(first_frame);
  packet.insert(packet.end(), 3, 0);
  packet.insert(packet.end(), second_frame.begin(), second_frame.end());
  uint32_t frame_sizes[8] = { 0 };
  int frame_count = -1;
  ASSERT_EQ(VPX_CODEC_OK, vp9_parse_superframe_index(
                              packet.data(), packet.size(), frame_sizes,
                              &frame_count, nullptr, nullptr));
  ASSERT_EQ(frame_count, 0);

  for (int encrypted = 0; encrypted <= 1; ++encrypted) {
    SCOPED_TRACE(encrypted ? "encrypted" : "plain");
    std::vector<uint8_t> input(packet);
    VP9Decoder decoder(dec_cfg, 0);
    RangeCheckedDecryptState decrypt_state = { nullptr, 0, 0, false };
    if (encrypted) {
      std::vector<uint8_t> ciphertext(input.size());
      encrypt_buffer(input.data(), ciphertext.data(), input.size(), 0);
      input.swap(ciphertext);
      decrypt_state.data = input.data();
      decrypt_state.size = input.size();
      vpx_decrypt_init decrypt_init = { range_checked_decrypt_cb,
                                        &decrypt_state };
      decoder.Control(VPXD_SET_DECRYPTOR, &decrypt_init);
    }
    ASSERT_EQ(VPX_CODEC_OK, decoder.DecodeFrame(input.data(), input.size()))
        << decoder.DecodeError();
    EXPECT_EQ(expected_md5, decoded_frame_md5(&decoder));
    EXPECT_EQ(decrypt_state.calls > 0, encrypted != 0);
    EXPECT_FALSE(decrypt_state.invalid_range);
  }
}

}  // namespace libvpx_test
