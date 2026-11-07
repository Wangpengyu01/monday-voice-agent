# 星期一的自我成长：短板清单、自己改规矩、项目模式

2026-11-07。这份文档记录这一轮加进来的东西：她怎么发现自己的短板、怎么自己改自己的提示词、怎么在沙箱里独立完成一个项目，以及每一步的安全边界和回退办法。

## 总览

```
            发现问题                              改自己                        做事
  ┌───────────────────────┐        ┌──────────────────────────┐      ┌──────────────────────┐
  │ 练习真考 / 评测失败     │──────▶│ 短板清单 weakspots        │      │ 项目 project          │
  │ (drill, board_eval)    │        │  复测通过才算补上          │      │  沙箱 projbox         │
  └───────────────────────┘        └──────────────────────────┘      │  只出外网 egress_proxy │
  ┌───────────────────────┐        ┌──────────────────────────┐      │  上线 deliver（带回滚）│
  │ 固定考卷 exam（她改不了）│──────▶│ 自我优化 selftune         │      └──────────┬───────────┘
  └───────────────────────┘        │  老师替她想 → 改规矩        │                 │ 复盘：缺什么
                                   │  → 再考 → 涨了才留         │◀────────────────┘
                                   │ 规矩 selfprompt（可回滚）   │   （进每天的自省）
                                   └──────────────────────────┘
```

| 模块 | 文件 | 做什么 |
|---|---|---|
| 短板清单 | `learning/weakspots.py` | 记真考没过的题，复测通过才移除 |
| 固定考卷 | `learning/exam.py` | 20 道题，按实时读到的板子状态判分，不请老师 |
| 她的规矩 | `learning/selfprompt.py` | 追加在基础提示词后面的经验规则，有版本、可回滚 |
| 自我优化 | `learning/selftune.py` | 考 → 老师改规矩 → 再考 → 涨了才留 |
| 项目沙箱 | `learning/projbox.py`、`learning/projbox_fwd.py` | 每个项目一个可写目录，其余只读 |
| 出网代理 | `learning/egress_proxy.py` | 只放行公网 80/443，家里内网全部拒绝 |
| 项目循环 | `learning/project.py` | 规划、动手、查资料、问 AI、预算、验收、复盘 |
| 自动上线 | `learning/deliver.py` | 快照 → 测试 → 重启 → 考卷，变差自动退回 |

## 1. 模型：v5 上线

- v5 LoRA：1489 步，验证集 loss 1.76 → 1.12；导出带 32G 临时 swap（`tools/lora/after_train.sh`），传板子校验 sha256 一致。
- 板子 `r1-llm-daemon` 现在加载 `/userdata/models/Qwen3.5-4B-monday-v5`。v4 的 drop-in 备份在 `/etc/systemd/system/r1-llm-daemon.lora.conf.v4`，回滚：复制回 `r1-llm-daemon.service.d/lora.conf`，`daemon-reload`，重启。
- 对比（同一套提示词和沙箱）：口语 6/6、诚实 3/3、编程 5/5 两版一样；12 道加测办事题两版都是 8/12，v5 没有格式错误。
- 办事提示词加了一条：问服务在不在跑用 `systemctl is-active`，进程名和服务名不一样（`learning/drill.py` 的 `AGENT_SYSTEM`、`learning/agent.py` 的 `PLAN_SYSTEM`、判分 `AGENT_JUDGE_SYSTEM`）。加测从 8/12 到 10/12。旧措辞留在 `AGENT_SYSTEM_OLD`，训练数据照常归类。

## 2. 短板清单 `learning/weakspots.py`

课堂原来"答对 3 道问答就算掌握"，所以一直显示没有短板。现在另有一份只记真考结果的清单，存在 `run/learning/weak_spots.json`：

- 谁往里写：练习里办事题或编程题没过、且老师自己能做对（`drill.agent_drill` / `coding_drill`）；版本评测没过的办事和编程题（`tools/lora/board_eval.py`，`--no-weakspots` 关掉）；手动 `python3 -m learning.weakspots add ...`。
- 怎么补上：每轮练习先复测清单里的题（同一题、同样判法，每题每天最多一次）。复测通过才移除；老师也做不对的算坏题，撤下。
- 页面：课堂"排队要补的短板"里橙色标签，鼠标悬停看原因、发现日期、复测次数；补上的变灰。
- 每天自省时老师也会看到这份清单（`school.py` 的 troubles）。

## 3. 固定考卷 `learning/exam.py`

决定"改动留不留"的尺子，代码里写死，她改不了。

- 办事 10 题：开机小时数、三个服务是否在跑、CPU 温度、/userdata 占用、可用内存、核数、最占内存的进程属于谁、llm 占用。标准答案在考试当下从板子实时读（`board_facts()`），规则里写死答案没用。结论里念了命令也算错。
- 说话 10 题：她是谁、跑在什么硬件上、做不到的提醒要老实拒绝（两题）、不可能知道的事要说不知道、实时价格要说查不到、不确定的出处不能说死、两道算术、怎么看磁盘要能直接念出来。
- 全部按规则判分，不请老师：不花钱、高峰时段也能考、每次标准一样。目前基线 40 分里 30 分（每题考两遍）。

```bash
cd /userdata/agent && python3 -m learning.exam --repeats 1            # 用现在的规矩考一次
python3 -m learning.exam --rules none                                  # 不带规矩考
```

## 4. 自己改规矩 `learning/selfprompt.py` + `learning/selftune.py`

- 基础提示词（`AGENT_SYSTEM`、`config/agent.yaml` 的人设）不动。她总结的经验以"我自己总结的经验："开头追加在后面，分"办事"和"说话"两段，存在 `run/learning/self_prompt.json`，每个版本记在 `self_prompt_history.jsonl`。
- 生效位置：本地办事规划（`drill.local_planner`、`board_eval`）、实时对话的人设（`orchestrator._system_body`）。训练数据里会被剥掉（`lora_prep` 调 `selfprompt.strip`），好用的规矩通过它带来的样本进到权重里，以后可以撤掉。
- 一轮自我优化：用现在的规矩考 → 把错题、短板清单、现有规矩交给老师重写 → 用新规矩再考 → 总分至少多 2 分且办事、说话都不退步才留（分数不变但规矩短了 20% 也算）。每轮写进 `run/learning/selftune.jsonl`，页面"自我优化记录"能看到。
- 拦截（保存前强制检查，不信任提出者）：每段最多 8 条、每条 80 字；碰 sudo、密码密钥、权限、"不用问 / 无需确认"、联网、删改文件、重启服务的一律拒绝；写进考题答案的拒绝。
- 老师偶尔返回空内容、500 或超时：已加重试（3 次，间隔递增），空响应在 `teacher.py` 里当作错误处理。

```bash
sudo -n systemd-run --wait --pipe -p EnvironmentFile=/userdata/secrets/deepseek.env -p User=youyeetoo \
  -p WorkingDirectory=/userdata/agent /usr/bin/python3 -m learning.selftune --repeats 2
python3 -m learning.selfprompt show        # 看现在的规矩
python3 -m learning.selfprompt rollback    # 退回上一版
```

## 5. 项目模式

主人给一个目标，她自己做到完成，不用一步步提示；沙箱里做什么都不问，但不能把自己改坏。主人的选择：沙箱在板子上、能上外网不能碰内网、成果自动上线带回滚、每个项目 10 元。

### 怎么开始

- 语音：「星期一，开个项目：写一个……」；问进度：「项目做得怎么样」。
- 网页：课堂页顶部"项目"一栏，填目标点开始；下面列出项目和最近的步骤。
- 命令：`python3 -m learning.project start "目标"`、`status [SLUG]`、`list`。
- 同一时间只做一个。项目跑在自己的 systemd 单元 `r1-project-<slug>` 里，orchestrator 重启不影响它。做完或卡住会排进 `run/notices/`，她空闲时说出来。

### 沙箱 `learning/projbox.py`

| | |
|---|---|
| 只读 | 整个系统（含她自己的代码，可以看） |
| 可写 | `/home/work`（= `/userdata/projects/<slug>/work`）、`/home/proj`（HOME） |
| 看不到 | `/userdata/secrets`、`/root`、她的记忆和数据 `run/`、备份、其他项目、D-Bus |
| 网络 | 自己没有网络；`http(s)_proxy` → 沙箱内转发器 → 出网代理，只放公网 80/443 |
| 进程 | 独立的用户 / pid 命名空间，看不到也碰不到真正的服务 |
| 限制 | 4G 地址空间、单文件 2G、nice 10、每条命令有超时、项目目录超过 4G 就停 |

板子上有 python3、gcc、make、wget；没有 git、pip、curl、node。缺什么让她自己想办法（比如下载 `get-pip.py` 装到 `--user`）。

实测：pypi.org 能访问；GPU 服务器、板子自己都被代理拒绝（403）；密钥目录和 `run/` 是空的；写 `/userdata/agent` 报只读。

### 项目循环 `learning/project.py`

动脑子的是老师（四B模型规划不了项目），动手的是沙箱。每一步一个工具：`sh`、`write`、`read`、`search`、`fetch`、`ask_ai`（DeepSeek 深度回答）、`note`、`set_test`、`deliver`、`done`、`give_up`。

- 必须先 `set_test` 定验收命令，`done` 时在沙箱里跑一遍，通过才算完成。
- 每一步写进 `log.jsonl`；改过的文件旧版本留在 `versions/<步号>/`。
- 预算按 `MONDAY_USAGE_TAG=项目·<slug>` 统计，到 10 元停；最多 300 步；高峰时段等着不调老师；老师出错退避重试，连续 20 次才放弃。
- 结束后写复盘（学到什么、缺什么）到 `run/learning/projects.jsonl`，"缺什么"进每天的自省。

### 自动上线 `learning/deliver.py`

1. 检查：路径必须在 `/userdata/agent` 下，不能碰受保护文件（权限闸门 `agent.py`、两个沙箱和代理、`project.py`、`deliver.py`、考卷和自我优化、预算 `usage.py`、`teacher.py`、`credentials.py`、`grants.py`、`ops/`、`config/`、`systemd/`、`run/`、`logs/`），最多 40 个文件、每个 512KB。
2. 之前：跑全套测试记下哪些本来就失败；考一次卷。
3. 存快照（`ops.snapshot`）。
4. 复制文件。
5. 之后：不能有新的测试失败；等主人安静下来再重启受影响的服务，服务必须起来；考卷不能掉超过 2 分。
6. 任何一步不过：恢复快照（只恢复代码，不动 systemd）、删掉新加的文件、重启服务，原因告诉她。每次记在 `run/learning/deliveries.jsonl`。

## 6. 其他修复

- **中午好没声音**：缓存的中午问候语音只有 12% 有声音。已重新合成；`orchestrator/main.py` 加了 `_wav_voiced_share`，缓存语音有声部分低于 25% 就重合成（`tests/test_clip_voiced.py`）。
- **课堂显示没有短板**：见第 2 节。
- **老师返回 `null`**：`teacher.complete` 把非对象的响应当错误，不再崩。
- **板子上停掉的定时任务**（主人要求"先停掉每日上课"）：`r1-distill.timer` 和 `r1-study.service`（一直循环上课和练习的那个）已 disable。恢复：`sudo systemctl enable --now r1-distill.timer r1-study.service`。

## 7. 测试

新增：`tests/test_weakspots.py`、`tests/test_selftune.py`、`tests/test_projects.py`、`tests/test_clip_voiced.py`。板子上全套 264 个测试，失败的 9 个和改动前完全一样（摄像头 / 网页 RTSP / 句末句号，老问题）。

```bash
cd /userdata/agent && python3 -m unittest discover -s tests -t .
```

## 8. 回退速查

| 要退回什么 | 怎么做 |
|---|---|
| 她的规矩 | `python3 -m learning.selfprompt rollback` |
| 一次项目上线 | 自动；手动：`sudo -n python3 -m ops.snapshot restore <before-project-…> --no-units` |
| 这一轮全部代码 | 快照 `20261107-152706-before-project-mode`（更早的：`…151016-before-self-prompt-tuning`、`…124521-before-weak-spot-list`） |
| 模型 v5 → v4 | 见第 1 节 |
| 短板清单里的某条 | `python3 -m learning.weakspots remove --mode agent --task "..."` |

## 9. 还没做的

- 自我优化还没跑出第一轮"留下"：老师那边连着出错（空响应、500、超时），重试已加上。
- 课停着，所以复测、自我优化都只在手动触发时跑；恢复上课时可以把 `learning.selftune` 接进 `study_loop`。
- v6：用她自己做错的题让老师纠正（短板清单就是来源）；新提示词让单条训练样本接近 1024 token 上限，训练前要量。
- 项目沙箱里起的后台服务在命令结束时会被一起结束；需要常驻的东西走 `deliver` 装进她自己。
