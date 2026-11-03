# 星期一 / M.O.N.D.A.Y. — 接力文档

> 读完这一篇就能接着干。详细背景见 [README.md](README.md)、[PROGRESS.md](PROGRESS.md)、[agent/README.md](agent/README.md)。
> 本文记录的是**当前实际在板子上运行的状态**（从「小揽 + InternVL」基线演进到「星期一 + Qwen3.5-4B」）。

## 这是什么

youyeetoo R1（**RK3588S** 主控 + **RK1828** 协处理器）上的本地语音智能体，名字叫**星期一 / M.O.N.D.A.Y.**。
目标是让它像个人：本地听说、会学、能操作家里的设备。除了「问老师」会调用 DeepSeek，其余全部在板子本地运行。

- 板子地址：`10.0.0.177`（登录用户 `youyeetoo`，用 SSH 公钥，**不要用密码/sshpass**）
- 语音链路：Silero VAD → Paraformer 流式 ASR → 本地 LLM → Qwen3-TTS → aplay

## 现在能做什么（均已在板子上验证）

### 1. 语音对话
- **本地 TTS**：Qwen3-TTS 跑在 RK1828，北京男声（`SPEAKER=beijing`）。曾经乱码/吞字，已修（见下「踩过的坑」）。
- **口头应答**：回答前先说「好嘞 / 我看看 / 哎呀」，按问题类型选（`orchestrator/quick_reply.py`）。
- **本地大脑**：Qwen3.5-4B（Rockchip 官方 RK1828 构建），ctx 768（和 TTS 同时跑放不下 1024），首字 ~0.7s。
- **开机音效 + 自检**：苹果式「咚—叮」(~1.5s)，开机播报存储/内存/温度（`orchestrator/selftest.py`）。
- **称呼可改**：说「以后叫我 X」，默认「主人」(`orchestrator/user_title.py`)。

### 2. 学习层 —「不会就问，问完记住」(`agent/learning/`)
路由顺序：**本地记忆 → 本地模型 → 老师模型(DeepSeek)**。
- 事实/查询/时效/个人信息/纠错 → 问老师；闲聊/安慰/关于星期一自己的问题 → 本地。
- 老师回答的同时附一张记忆卡 `【记忆】{...}`，被剥离出语音后存进 SQLite（FTS5 trigram 中文检索）。
- 换个说法也能命中记忆；说「不对…」会改正并存成教训；用户原话存成别名。
- 每轮老师问答按对话格式写进 `lora_data.jsonl`，攒够了拿去训练本地模型的 LoRA。
- 老师 = DeepSeek，OpenAI 兼容：`base_url=https://api.deepseek.com`，`model=deepseek-flash`，key 从环境变量 `DEEPSEEK_API_KEY` 读。

### 3. 设备发现与账号库 (`agent/devices/`)
- **扫描**：纯 stdlib（TCP 探测 + ARP + mDNS/avahi + SSDP/UPnP + 网页标题），不依赖 nmap。能认出路由器/Windows/Mac/NAS/手机/摄像头/打印机/智能家居等。
- **语音指令**：「扫描设备」「家里有哪些设备」「设备配对码」。
- **设备网页**：`http://10.0.0.177:8767`，仅局域网可访问，需先输入星期一念出的**一次性配对码**。可给设备起名、填账号密码。
- **加密账号库**：密码用 Fernet(AES-128-CBC+HMAC) 加密存本地（`run/devices/vault.db`），密钥在板子上 `R1_VAULT_KEY`。**绝不进模型提示词，绝不发云端**，`describe()` 只返回「哪台设备有哪类账号」不含明文。

## 架构速览

| 组件 | 位置 | 端口 / Socket |
|------|------|--------------|
| 编排器（主循环、对话、学习、设备指令） | `agent/orchestrator/main.py` | WS `:8765` |
| 控制台 / 聊天网页 | `agent/ui/` | HTTP `:8766` |
| 设备管理网页 | `agent/devices/web.py` | HTTP `:8767`（仅局域网） |
| 本地 LLM 守护 | `agent/daemon/`（C++） | `/tmp/r1-llm.sock` |
| TTS 守护 | `/userdata/qwen3tts/...`（板子上） | `/tmp/r1-tts.sock` |
| 学习层 | `agent/learning/{memory,teacher,brain}.py` | SQLite `run/learning/memory.db` |
| 设备层 | `agent/devices/{scanner,inventory,vault,voice,web}.py` | SQLite `run/devices/*.db` |

systemd 单元都在 `agent/systemd/`。部署脚本 `agent/scripts/board_apply_deploy.sh`（装单元、sysctl、自动生成保险库密钥）。

## 部署到板子

```bash
# 在仓库 agent/ 目录，rsync/scp 到板子 /userdata/agent/，然后：
sudo bash /userdata/agent/scripts/board_apply_deploy.sh
```

> ⚠ **不要**跑上游的 `install_autostart.sh`（会和现有配置冲突）。

## 密钥（都不在仓库里，需自己配）

| 密钥 | 位置（板子，root 600） | 说明 |
|------|----------------------|------|
| `DEEPSEEK_API_KEY` | `/userdata/secrets/deepseek.env` | 老师模型。没有它则全部走本地模型。 |
| `R1_VAULT_KEY` | `/userdata/secrets/vault.env` | 设备账号库密钥。部署脚本首次会自动生成。**丢了就解不开已存的密码**。 |

systemd 用 `EnvironmentFile=-/userdata/secrets/*.env` 加载（`-` = 文件不存在也不报错）。

## 测试

```bash
cd /userdata/agent        # 或本地 agent/
python3 -m unittest tests.test_learning tests.test_devices
```
学习层 + 设备层共 21 个测试。整套 `unittest discover` 还有 4 个 error 是上游遗留（vision/camera/ws），与本项目无关。

## 下一步 TODO

- [ ] **开关电脑**（进行中）：WoL 魔法包开机（板子无 wakeonlan，用纯 Python 发 UDP 广播）；paramiko(板子已有 2.9.3) + 库里账号远程关机。语音「打开/关掉 X」。
- [ ] **网页聊天**：`:8766` 已经能聊天并走 DeepSeek，待确认端到端体验，必要时美化。
- [ ] 课表 + 主动提醒（「下课了，回宿舍吗？」）。
- [ ] Tailscale 内网穿透，公网可达。
- [ ] 夜间整理记忆；本地答得没把握时自动再问一次老师（`brain.local_answer_unsure` 已有，未接线）。
- [ ] 把 `tools/` 下的一次性补丁脚本收进仓库；清理 TTS 调试代码。
- [ ] 有 GPU 后用 `lora_data.jsonl` 训练本地模型 LoRA。

## 踩过的坑（别重复）

- **TTS 乱码**不是量化问题，是 C++ demo 的 bug：embed dim 被写死 2048（0.6B 应为 1024）、请求间 KV cache 复用没清、尾部 off-by-one。排查要先和 PyTorch 数值逐行对比，别凭听感猜。
- **重启卡死**是 WiFi 驱动在 AP 模式挂住（旧 Hotspot profile），不是 sysctl。已把热点 autoconnect=false。
- **Qwen3.5-4B ctx 1024** 和 TTS 同时跑 OOM（MODEL_SETUP fail），降到 768。
- Shell heredoc 常把 `\n`/`\1` 吃掉，改用文件写补丁脚本再跑。
