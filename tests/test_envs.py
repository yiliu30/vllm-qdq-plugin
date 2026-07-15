import os
import unittest
from unittest import mock

from vllm_qdq_plugin import envs


class EnvTests(unittest.TestCase):
    def test_case_insensitive_choice_returns_canonical_value(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"VLLM_MARLIN_MOE_QDQ_MODE": "force_mxfp4"},
            clear=False,
        ):
            self.assertEqual(envs.VLLM_MARLIN_MOE_QDQ_MODE, "FORCE_MXFP4")

    def test_invalid_choice_raises(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"VLLM_MARLIN_MOE_QDQ_MODE": "bad_mode"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "Invalid value 'bad_mode'"):
                _ = envs.VLLM_MARLIN_MOE_QDQ_MODE

    def test_sparse_xpu_tile_envs_default_to_none(self) -> None:
        with mock.patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            self.assertIsNone(envs.SPARGE_QUERY_TILE_TOKENS)
            self.assertIsNone(envs.SPARGE_SPARSE_Q_BLOCK_TOKENS)
            self.assertIsNone(envs.SPARGE_SPARSE_K_BLOCK_TOKENS)

    def test_sparse_xpu_tile_envs_return_strings_when_set(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SPARGE_XPU_TENSOR_LAYOUT": "hnd",
                "SPARGE_QUERY_TILE_TOKENS": "256",
                "SPARGE_SPARSE_Q_BLOCK_TOKENS": "256",
                "SPARGE_SPARSE_K_BLOCK_TOKENS": "64",
            },
            clear=True,
        ):
            self.assertEqual(envs.SPARGE_XPU_TENSOR_LAYOUT, "HND")
            self.assertEqual(envs.SPARGE_QUERY_TILE_TOKENS, "256")
            self.assertEqual(envs.SPARGE_SPARSE_Q_BLOCK_TOKENS, "256")
            self.assertEqual(envs.SPARGE_SPARSE_K_BLOCK_TOKENS, "64")

    def test_sparse_xpu_tensor_layout_defaults_to_nhd(self) -> None:
        with mock.patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            self.assertEqual(envs.SPARGE_XPU_TENSOR_LAYOUT, "NHD")

    def test_sparse_dump_envs_defaults(self) -> None:
        with mock.patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            self.assertFalse(envs.SPARGE_DUMP_INPUTS)
            self.assertEqual(envs.SPARGE_DUMP_DIR, "/tmp/sparge_inputs")
            self.assertEqual(envs.SPARGE_DUMP_MAX, "1")
            self.assertEqual(envs.SPARGE_DUMP_START_INDEX, "0")
