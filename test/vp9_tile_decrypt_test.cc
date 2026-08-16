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

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/codec_factory.h"
#include "test/md5_helper.h"
#include "test/video_source.h"
#include "vpx/vp8cx.h"

namespace {

const uint8_t kTestKey[16] = { 0x01, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
                               0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0xf0 };

void encrypt_buffer(const uint8_t *src, uint8_t *dst, size_t size,
                    size_t offset) {
  for (size_t i = 0; i < size; ++i) {
    dst[i] = src[i] ^ kTestKey[(offset + i) & 15];
  }
}

struct CheckedDecryptState {
  const uint8_t *data;
  size_t size;
  int calls;
  bool invalid_range;
};

void checked_decrypt_cb(void *decrypt_state, const uint8_t *input,
                        uint8_t *output, int count) {
  CheckedDecryptState *const state =
      static_cast<CheckedDecryptState *>(decrypt_state);
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
  encrypt_buffer(input, output, size, offset);
}

}  // namespace

namespace libvpx_test {

TEST(TestDecrypt, GeneratedMultiTileFrameDecrypts) {
  RandomVideoSource video;
  video.SetSize(512, 64);
  video.set_limit(1);
  video.Begin();

  vpx_codec_enc_cfg_t enc_cfg;
  ASSERT_EQ(VPX_CODEC_OK, kVP9.DefaultEncoderConfig(&enc_cfg, 0));
  enc_cfg.g_lag_in_frames = 0;
  enc_cfg.rc_target_bitrate = 200;
  TwopassStatsStore stats;
  std::unique_ptr<Encoder> encoder(
      kVP9.CreateEncoder(enc_cfg, VPX_DL_REALTIME, 0, &stats));
  ASSERT_NE(encoder, nullptr);
  encoder->InitEncoder(&video);
  encoder->Control(VP9E_SET_TILE_COLUMNS, 1);
  encoder->Control(VP8E_SET_CPUUSED, 8);
  encoder->EncodeFrame(&video, VPX_EFLAG_FORCE_KF);

  std::vector<uint8_t> frame;
  CxDataIterator packets = encoder->GetCxData();
  while (const vpx_codec_cx_pkt_t *pkt = packets.Next()) {
    if (pkt->kind != VPX_CODEC_CX_FRAME_PKT) continue;
    const uint8_t *const data =
        static_cast<const uint8_t *>(pkt->data.frame.buf);
    frame.assign(data, data + pkt->data.frame.sz);
    break;
  }
  ASSERT_FALSE(frame.empty());

  const vpx_image_t *const preview = encoder->GetPreviewFrame();
  ASSERT_NE(preview, nullptr);
  MD5 expected_md5;
  expected_md5.Add(preview);
  const std::string expected = expected_md5.Get();

  std::vector<uint8_t> encrypted(frame.size());
  encrypt_buffer(frame.data(), encrypted.data(), frame.size(), 0);
  CheckedDecryptState checked = { encrypted.data(), encrypted.size(), 0,
                                  false };
  vpx_decrypt_init decrypt_init = { checked_decrypt_cb, &checked };
  vpx_codec_dec_cfg_t dec_cfg = vpx_codec_dec_cfg_t();
  dec_cfg.threads = 1;
  VP9Decoder decoder(dec_cfg, 0);
  decoder.Control(VPXD_SET_DECRYPTOR, &decrypt_init);
  ASSERT_EQ(VPX_CODEC_OK,
            decoder.DecodeFrame(encrypted.data(), encrypted.size()))
      << decoder.DecodeError();
  DxDataIterator frames = decoder.GetDxData();
  const vpx_image_t *const decoded = frames.Next();
  ASSERT_NE(decoded, nullptr);
  MD5 decoded_md5;
  decoded_md5.Add(decoded);
  EXPECT_STREQ(expected.c_str(), decoded_md5.Get());
  EXPECT_EQ(frames.Next(), nullptr);
  EXPECT_GT(checked.calls, 0);
  EXPECT_FALSE(checked.invalid_range);
}

}  // namespace libvpx_test
