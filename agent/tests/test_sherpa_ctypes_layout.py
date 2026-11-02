"""ctypes structs must match sherpa-onnx v1.12.8 c-api.h byte for byte (64-bit ABI)."""
import ctypes
import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from asr.sherpa_streaming_asr import _OnlineModelConfig, _OnlineRecognizerConfig  # noqa: E402


@unittest.skipUnless(ctypes.sizeof(ctypes.c_void_p) == 8, "64-bit ABI only")
class SherpaCtypesLayoutTests(unittest.TestCase):
    def test_online_model_config_matches_c_api(self) -> None:
        self.assertEqual(_OnlineModelConfig.tokens.offset, 48)
        self.assertEqual(_OnlineModelConfig.num_threads.offset, 56)
        self.assertEqual(_OnlineModelConfig.provider.offset, 64)
        self.assertEqual(_OnlineModelConfig.debug.offset, 72)
        self.assertEqual(_OnlineModelConfig.nemo_ctc.offset, 120)
        self.assertEqual(ctypes.sizeof(_OnlineModelConfig), 128)

    def test_online_recognizer_config_fields_after_model_config_line_up(self) -> None:
        self.assertEqual(_OnlineRecognizerConfig.model_config.offset, 8)
        self.assertEqual(_OnlineRecognizerConfig.decoding_method.offset, 136)
        self.assertEqual(_OnlineRecognizerConfig.enable_endpoint.offset, 148)


if __name__ == "__main__":
    unittest.main()
