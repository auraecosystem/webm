/*
 *  Copyright (c) 2026 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdint.h>
#include <string.h>

#include <vector>

#include "gtest/gtest.h"

#include "test/codec_factory.h"
#include "test/video_source.h"
#include "vpx/vp8cx.h"

namespace {

const uint8_t kPartitionTestKey[16] = { 0x01, 0x12, 0x23, 0x34, 0x45, 0x56,
                                        0x67, 0x78, 0x89, 0x9a, 0xab, 0xbc,
                                        0xcd, 0xde, 0xef, 0xf0 };

struct PartitionDecryptState {
  const uint8_t *data;
  size_t size;
  int calls;
  bool invalid_range;
};

void partition_decrypt_cb(void *decrypt_state, const uint8_t *input,
                          uint8_t *output, int count) {
  PartitionDecryptState *const state =
      static_cast<PartitionDecryptState *>(decrypt_state);
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

  for (size_t i = 0; i < size; ++i) {
    output[i] = input[i] ^ kPartitionTestKey[(offset + i) & 15];
  }
}

class VP8EncryptedPartitionsTest : public ::libvpx_test::EncoderTest,
                                   public ::testing::Test {
 protected:
  VP8EncryptedPartitionsTest()
      : EncoderTest(&::libvpx_test::kVP8), encoder_configured_(false),
        frame_seen_(false), decrypt_state_() {
    memset(&modified_pkt_, 0, sizeof(modified_pkt_));
  }
  ~VP8EncryptedPartitionsTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource * /*video*/,
                          ::libvpx_test::Encoder *encoder) override {
    if (!encoder_configured_) {
      encoder->Control(VP8E_SET_TOKEN_PARTITIONS,
                       static_cast<int>(VP8_FOUR_TOKENPARTITION));
      encoder_configured_ = true;
    }
  }

  const vpx_codec_cx_pkt_t *MutateEncoderOutputHook(
      const vpx_codec_cx_pkt_t *pkt) override {
    if (pkt->kind != VPX_CODEC_CX_FRAME_PKT) return pkt;

    const uint8_t *const clear =
        static_cast<const uint8_t *>(pkt->data.frame.buf);
    const size_t frame_size = pkt->data.frame.sz;

    encrypted_.resize(frame_size);
    for (size_t i = 0; i < frame_size; ++i) {
      encrypted_[i] = clear[i] ^ kPartitionTestKey[i & 15];
    }

    decrypt_state_.data = encrypted_.data();
    decrypt_state_.size = encrypted_.size();
    decrypt_state_.calls = 0;
    decrypt_state_.invalid_range = false;

    modified_pkt_ = *pkt;
    modified_pkt_.data.frame.buf = encrypted_.data();
    frame_seen_ = true;
    return &modified_pkt_;
  }

  void PreDecodeFrameHook(::libvpx_test::VideoSource * /*video*/,
                          ::libvpx_test::Decoder *decoder) override {
    ASSERT_TRUE(frame_seen_);
    vpx_decrypt_init decrypt_init = { partition_decrypt_cb, &decrypt_state_ };
    decoder->Control(VPXD_SET_DECRYPTOR, &decrypt_init);
  }

  bool encoder_configured_;
  bool frame_seen_;
  std::vector<uint8_t> encrypted_;
  vpx_codec_cx_pkt_t modified_pkt_;
  PartitionDecryptState decrypt_state_;
};

TEST_F(VP8EncryptedPartitionsTest, EncryptedFourTokenPartitionRoundTrip) {
  ::libvpx_test::RandomVideoSource video;
  video.set_limit(1);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  ASSERT_TRUE(frame_seen_);
  EXPECT_GT(decrypt_state_.calls, 0);
  EXPECT_FALSE(decrypt_state_.invalid_range);
}

}  // namespace
