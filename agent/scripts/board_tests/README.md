# 板端实机测试脚本

在开发板上直接运行（脚本内写死了 `/userdata/agent` 路径），用 TTS 合成测试语音再回放给麦克风，验证整条语音链路。

| 脚本 | 用途 |
|---|---|
| `run_stress_test.py` | 声学压力测试：喇叭对着麦克风时不自打断、"停"能打断、"继续"能接着说 |
| `run_live_test.py` | 合成几句测试语音，回放并观察对话日志 |
| `run_vision_live_test.py` | 语音问"看看桌上放了什么"，验证视觉问答链路 |
| `validate_dt.py` | 语音问"今天是星期几"，验证日期时间回答 |
| `test_llm.py` | 直接通过 `/tmp/r1-llm.sock` 向 llm_daemon 提问 |

示例：`python3 /userdata/agent/scripts/board_tests/run_stress_test.py`
