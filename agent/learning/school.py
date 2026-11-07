"""Daily school: 星期一 asks itself who it is, what it is for and what it lacks, then has class with the
teacher (DeepSeek) the way a student does - lecture, exercises, graded homework - so the small local
model comes to *understand* code, not just store answers.

One lesson per run (r1-school.timer, every half hour - "一直上课" - skipped while the owner is talking;
the first lesson of a day starts with the self-review):
  1. self-review  - identity + purpose + what it can do now (skill packs, learned skills, tools) +
                    what went wrong recently (failed tasks, wrong answers) -> the teacher names the
                    gaps and today's gap lesson.
  2. class        - two lessons: the next topic of the programming syllabus (Python, Linux tools,
                    git, ssh, systemd, docker ...) and the gap lesson. The teacher explains with a short
                    example and sets exercises: "read" (what does this code do / print) and "write"
                    (a small function, checked by the teacher's asserts).
  3. homework     - the local model answers; written code runs in a sandbox (checked first: no os,
                    subprocess, network or file writes; temp dir, CPU/memory limits, timeout).
  4. marking      - the teacher grades with the run results, explains each mistake, gives the right
                    answer.
  5. remembering  - class notes (run/learning/school/), key points as memory cards, every
                    question -> right explanation/code as LoRA training data; a passed topic moves the
                    syllabus on, a failed one comes back tomorrow from another angle. A gap that is a
                    missing *ability* gets a draft skill pack (run/skills/, always needs approval to run).

    cd /userdata/agent && python3 -m learning.school [--force] [--topic "Python 正则表达式"] [--full-day]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from learning import now  # noqa: E402

LOG = logging.getLogger("school")
REVIEW_DAYS = 14          # a mastered topic is skipped this long, then reviewed once
STUDENT_MAX_TOKENS = 360  # room for a small function; the prompt is kept short to fit the 768 context
# Coding mode: the prompt the local model gets when it writes code itself (class now, own scripts later).
# Class samples are trained under THIS prompt, not the voice persona, so she keeps writing real code
# here while her spoken replies stay spoken (learning/lora_prep.py keeps the two modes apart).
CODING_SYSTEM = "你是学生星期一，正在上编程课。认真读题，答案简洁准确，写代码就只写代码。"

SYLLABUS = [
    "Python 基础：变量、列表、字典、循环和函数",
    "Python 字符串处理和格式化",
    "Python 读写文本文件和 with 语句",
    "Python 的 json：读写和遍历",
    "Python 正则表达式 re",
    "Python 异常处理 try/except",
    "Python 列表推导式、sorted 和 lambda",
    "Python 类和 dataclass",
    "Python 模块、import 和 if __name__ == '__main__'",
    "Python 命令行参数 argparse",
    "Python 的 logging 日志",
    "Python 时间日期 datetime",
    "Python 用 subprocess 运行命令并读输出",
    "Python 用 pathlib 处理路径",
    "Python 的 sqlite3 数据库",
    "Python 网络请求 urllib / requests",
    "Python asyncio 基础",
    "Python 单元测试 unittest",
    "bash 基础：变量、管道、重定向、if 和 for",
    "grep、sed、awk 处理文本",
    "Linux 进程和资源：ps、top、free、df",
    "systemd：systemctl 和 journalctl",
    "ssh、scp、rsync 远程操作",
    "git 基础：clone、status、diff、commit、log",
    "pip 和 venv 管理 Python 环境",
    "docker 基础：镜像、容器、日志",
    "网络排查：ip、ping、ss、curl",
    "读懂别人的 Python 项目：从入口文件开始",
]

SELF_SYSTEM = (
    "你是语音助手「星期一」的老师。星期一的定位是全栈工程师型的私人助理：什么都会一点，能自己写程序、"
    "升级自己、认识自己（它的代码和板子硬件），最终目标是读懂主人的日常生活、主动帮主人把日子过好。"
    "它要每天先认清自己，再像学生一样上课。根据它给出的自我介绍、对主人的了解、现在会的能力和最近做砸的事，"
    "帮它回答：我是谁、我要干什么、我缺什么（包括为了更懂主人还缺哪些信息或能力），并定下今天补短板的那一课。"
    "缺口要具体（比如「不会把文件传到别的机器」「看不懂 Python 的异常堆栈」「不知道主人的作息」），优先选主人最可能用到的。"
    '只输出 JSON：{"who_am_i": "一句话", "purpose": "一句话", "gaps": [{"gap": "...", "why": "...", '
    '"kind": "knowledge|coding|ability"}], "lesson_topic": "今天补短板这一课的题目", '
    '"ability_skill": {"name": "英文短名", "description": "这个能力做什么", "need": "要写的脚本做什么"} 或 null}'
)
LECTURE_SYSTEM = (
    "你在给一个四B小模型（学生「星期一」）上编程课，它跑在一块 Linux 板子上，回答会被朗读。讲课要短而准："
    "讲清一个核心概念，配一段不超过二十五行的示例代码（Python 优先；讲 Linux 工具就给命令示例），"
    "再出三道练习：两道 read（看代码回答它做什么或输出什么），一道 write（写一个小函数，函数名固定，"
    "并给出用来检验的 assert 语句；只能用标准库，不能读写文件、不能联网、不能 import os/subprocess/socket）。"
    '只输出 JSON：{"title": "...", "explain": "三五句讲解", "example": "示例代码", "key_points": ["要点"...], '
    '"exercises": [{"kind": "read", "code": "代码", "question": "问题", "answer": "标准答案"}, '
    '{"kind": "write", "question": "题目（写明函数名和参数）", "tests": "assert 语句，多行", "answer": "参考代码"}]}'
)
SELF_LECTURE_SYSTEM = (
    "你在给一个四B小模型（学生「星期一」）上「认识自己」课：下面的材料是它自己的源代码，或者它所在板子"
    "（youyeetoo R1，RK3588S + RK1828）的硬件勘察结果。只根据材料讲，材料里没有的不要编。"
    "用第二人称讲（「你的……」）：这段代码/这个硬件是干什么的、在整个系统里处在哪一环、出了问题从哪里查。"
    "讲解三五句，口语化；示例给材料里最关键的几行（不超过十五行）。再出三道 read 题考它对材料的理解，"
    "题目里的 code 字段放材料里的相关几行，标准答案要能从材料里找到依据。硬件课可以出一道算 GPIO 编号这类的题。"
    '只输出 JSON：{"title": "...", "explain": "三五句讲解", "example": "关键几行", "key_points": ["要点"...], '
    '"exercises": [{"kind": "read", "code": "相关几行", "question": "问题", "answer": "标准答案"}]}'
)
GRADE_SYSTEM = (
    "你在批改学生「星期一」（四B小模型）的编程作业。给每道题判 pass 或 fail；write 题以运行结果为准，"
    "运行通过才算 pass。每道题写一句点评，fail 的要讲清错在哪、为什么，并给出正确的答案或代码。"
    '只输出 JSON：{"items": [{"i": 序号, "pass": true或false, "comment": "点评", "correct": "正确答案或代码"}], '
    '"summary": "一句话总评", "mastered": true或false}'
)
SKILL_SYSTEM = (
    "为语音助手「星期一」写一个技能包，让它学会一种新能力。它跑在 Ubuntu 板子上（用户 youyeetoo，工作目录 /userdata/agent），"
    "执行脚本前会先问主人。脚本用 bash 或 python3（只用标准库），非交互，出错要给中文提示，"
    "不碰 /userdata/secrets、私钥和账号库，不删除、不关机、不改系统配置。"
    '只输出 JSON：{"name": "英文短名", "description": "一句话", "tags": "逗号分隔关键词", '
    '"skill_md": "SKILL.md 正文（做法、注意事项，引用脚本路径 /userdata/agent/run/skills/<name>/scripts/<文件名>）", '
    '"script_name": "文件名.sh 或 .py", "script": "脚本内容"}'
)

_BANNED = re.compile(r"\b(import\s+(os|sys|subprocess|socket|shutil|ctypes|multiprocessing|threading|urllib|http|requests|pathlib|pickle|importlib)"
                     r"|from\s+(os|sys|subprocess|socket|shutil|ctypes|multiprocessing|urllib|http|pathlib|importlib)\b"
                     r"|__import__|\beval\s*\(|\bexec\s*\(|\bopen\s*\(|\bcompile\s*\(|globals\s*\(|getattr\s*\(\s*__builtins__)")
_SCRIPT_BANNED = re.compile(r"/userdata/secrets|vault|\.ssh/id_|private[_ ]?key|\brm\s+-rf?\s+/|\bmkfs|\bdd\s+if=|"
                            r"\b(shutdown|reboot|poweroff|halt)\b|\bpasswd\b|curl[^|]*\|\s*(ba)?sh", re.I)


@dataclass
class Lesson:
    topic: str
    kind: str  # syllabus | gap
    title: str = ""
    explain: str = ""
    example: str = ""
    key_points: list[str] = field(default_factory=list)
    exercises: list[dict] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    grades: list[dict] = field(default_factory=list)
    summary: str = ""
    mastered: bool = False


def _json(text: str) -> dict:
    i, j = (text or "").find("{"), (text or "").rfind("}")
    try:
        obj = json.loads(text[i:j + 1]) if 0 <= i < j else {}
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _code_of(text: str) -> str:
    """The code inside a ```python block, or the whole answer."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return (m.group(1) if m else text or "").strip()


# ---------------------------------------------------------------- sandbox
def check_code(code: str) -> str:
    """'' if the student's code may run, else why not."""
    m = _BANNED.search(code)
    if m:
        return f"代码里有不允许的用法：{m.group(0)}"
    if len(code) > 4000:
        return "代码太长"
    return ""


def run_sandboxed(code: str, tests: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Run code + asserts in a throwaway dir with CPU/memory limits; -> (passed, output)."""
    why = check_code(code)
    if why:
        return False, why
    program = code + "\n\n" + (tests or "") + "\nprint('ALL_TESTS_PASSED')\n"
    preexec = None
    if sys.platform != "win32":
        def preexec():  # noqa: E306 - limits apply only to the child
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (int(timeout), int(timeout)))
            resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    with tempfile.TemporaryDirectory() as td:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", program], cwd=td, capture_output=True, text=True,
                               timeout=timeout + 1, env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"},
                               preexec_fn=preexec, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False, "运行超时"
    out = (r.stdout + r.stderr).strip()
    ok = r.returncode == 0 and "ALL_TESTS_PASSED" in r.stdout
    return ok, out.replace("ALL_TESTS_PASSED", "").strip()[-600:] or ("全部通过" if ok else f"返回码 {r.returncode}")


def check_script(text: str) -> str:
    m = _SCRIPT_BANNED.search(text or "")
    return f"脚本里有不允许的内容：{m.group(0)}" if m else ""


# ---------------------------------------------------------------- the day
@dataclass
class SchoolState:
    path: Path
    index: int = 0
    history: list[dict] = field(default_factory=list)
    day: str = ""                                          # date of the last self-review
    review: dict = field(default_factory=dict)             # that review (who_am_i, purpose, gaps)
    pending: list[str] = field(default_factory=list)       # gap lessons still to give (today's + failed ones)
    sessions: int = 0                                      # lessons given so far (rotates syllabus / self / gap)
    self_index: int = 0                                    # next 认识自己 lesson (hardware first, then her code)
    self_tries: int = 0                                    # attempts at the current self lesson
    mastered: dict[str, str] = field(default_factory=dict)  # topic -> day it was last mastered

    @classmethod
    def load(cls, path: Path) -> "SchoolState":
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            st = cls(path, int(d.get("index", 0)), list(d.get("history", [])), str(d.get("day", "")),
                     dict(d.get("review", {})), list(d.get("pending", [])), int(d.get("sessions", 0)),
                     int(d.get("self_index", 0)), int(d.get("self_tries", 0)), dict(d.get("mastered", {})))
        except (OSError, ValueError):
            return cls(path)
        if "mastered" not in d:   # older state: what the recent history shows as mastered
            for h in st.history:
                if h.get("mastered") and h.get("topic"):
                    st.mastered[h["topic"]] = str(h.get("day") or "")
        return st

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"index": self.index, "history": self.history[-300:], "day": self.day,
                                         "review": self.review, "pending": self.pending[:20], "sessions": self.sessions,
                                         "self_index": self.self_index, "self_tries": self.self_tries,
                                         "mastered": self.mastered},
                                        ensure_ascii=False, indent=1), encoding="utf-8")

    def next_topic(self) -> str:
        return SYLLABUS[self.index % len(SYLLABUS)]

    def fresh(self, topic: str, day: str) -> bool:
        """Mastered less than REVIEW_DAYS ago: teaching it again only costs the teacher money."""
        when = self.mastered.get(topic)
        if not when:
            return False
        try:
            age = (dt.date.fromisoformat(day) - dt.date.fromisoformat(when)).days
        except ValueError:
            return False
        return age < REVIEW_DAYS

    def due_syllabus(self, day: str) -> str | None:
        """The next syllabus topic not freshly mastered (index moves to it), or None if all are."""
        for k in range(len(SYLLABUS)):
            topic = SYLLABUS[(self.index + k) % len(SYLLABUS)]
            if not self.fresh(topic, day):
                self.index += k
                return topic
        return None

    def due_self(self, selfs: list[tuple[str, str]], day: str) -> tuple[str, str] | None:
        for k in range(len(selfs)):
            topic, material = selfs[(self.self_index + k) % len(selfs)]
            if not self.fresh(topic, day):
                if k:
                    self.self_index, self.self_tries = self.self_index + k, 0
                return topic, material
        return None


def self_review(teacher, *, identity: str, abilities: str, troubles: str) -> dict:
    return _json(teacher.complete(SELF_SYSTEM, f"我的自我介绍：{identity}\n\n我现在会的：\n{abilities}\n\n最近做砸的：\n{troubles or '没有记录'}",
                                  json_mode=True, max_tokens=900))


def teach(teacher, topic: str, kind: str, local_answer: Callable[[str], str],
          run: Callable[[str, str], tuple[bool, str]] = run_sandboxed, material: str = "") -> Lesson:
    now.step("老师在备课讲课")
    if material:   # 认识自己: the teacher teaches from her own code / hardware survey
        lec =_json(teacher.complete(SELF_LECTURE_SYSTEM, f"这节课：{topic}\n\n材料：\n{material[:6000]}",
                                     json_mode=True, max_tokens=2000))
    else:
        lec = _json(teacher.complete(LECTURE_SYSTEM, f"这节课：{topic}", json_mode=True, max_tokens=2000))
    lesson = Lesson(topic, kind, str(lec.get("title") or topic), str(lec.get("explain") or ""), str(lec.get("example") or ""),
                    [str(x) for x in lec.get("key_points") or []],
                    [e for e in lec.get("exercises") or [] if isinstance(e, dict) and e.get("question")][:4])
    notes = f"今天学：{lesson.title}\n{lesson.explain}\n示例：\n{lesson.example}"
    for n, ex in enumerate(lesson.exercises, 1):
        now.step(f"做第 {n}/{len(lesson.exercises)} 题：{ex['question'][:40]}")
        if ex.get("kind") == "write":
            prompt = f"{notes[:300]}\n\n题目：{ex['question']}\n只写 Python 代码，放在 ```python 代码块里。"
            ans = local_answer(prompt) or ""
            ok, out = run(_code_of(ans), str(ex.get("tests") or ""))
            lesson.runs.append(("通过" if ok else "没通过") + f"：{out}")
        else:
            ans = local_answer(f"{notes[:200]}\n\n看这段代码：\n{ex.get('code', '')}\n问题：{ex['question']}\n用一两句话回答。") or ""
            lesson.runs.append("")
        lesson.answers.append(ans)
    now.step("老师在批改")
    listing ="\n\n".join(
        f"{i}. [{ex.get('kind')}] {ex['question']}\n" + (f"代码：\n{ex.get('code')}\n" if ex.get("code") else "")
        + f"学生答：{a[:800]}\n" + (f"运行结果：{r}\n" if r else "") + f"参考答案：{str(ex.get('answer', ''))[:500]}"
        for i, (ex, a, r) in enumerate(zip(lesson.exercises, lesson.answers, lesson.runs)))
    g = _json(teacher.complete(GRADE_SYSTEM, f"课题：{lesson.title}\n{listing}", json_mode=True, max_tokens=2000))
    by_i = {int(it.get("i", -1)): it for it in g.get("items") or [] if isinstance(it, dict)}
    lesson.grades = [by_i.get(i, {"pass": False, "comment": "老师没批到", "correct": ""}) for i in range(len(lesson.exercises))]
    passed = sum(1 for x in lesson.grades if x.get("pass"))
    lesson.mastered = bool(g.get("mastered")) and passed * 3 >= 2 * max(1, len(lesson.exercises))
    lesson.summary = str(g.get("summary") or f"{passed}/{len(lesson.exercises)} 道对")
    return lesson


def remember(lesson: Lesson, *, store, log_training: Callable[[str, str], None], notes_dir: Path, day: str) -> Path:
    """Class notes, memory cards for the key points, and question -> right answer as training data."""
    for i, p in enumerate(lesson.key_points[:6]):
        store.remember(f"{lesson.title}：要点{i + 1}", p, source="school")
    if lesson.kind == "self" and lesson.explain:   # so she can answer "你的 xx 是干什么的" by voice
        store.remember(lesson.topic.replace("认识自己 · ", "") + " 是做什么的", lesson.explain, source="self")
    log_training(f"给我讲讲{lesson.title}", f"{lesson.explain}\n例子：\n{lesson.example}")
    for ex, ans, grade in zip(lesson.exercises, lesson.answers, lesson.grades):
        right = str(grade.get("correct") or ex.get("answer") or "").strip()
        q = ex["question"] if ex.get("kind") == "write" else f"这段代码：\n{ex.get('code', '')}\n{ex['question']}"
        if right:
            log_training(q, right)
        if not grade.get("pass") and right:
            store.add_lesson(q[:200], ans[:300], right[:600])
    notes_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w]+", "-", lesson.title)[:40].strip("-") or "lesson"
    path = notes_dir / f"{day}-{slug}.md"
    label = {"syllabus": "课程表", "gap": "补短板", "self": "认识自己"}.get(lesson.kind, lesson.kind)
    parts = [f"# {lesson.title}（{label}）", "", lesson.explain, "",
             "```", lesson.example, "```", "", "## 要点", *[f"- {p}" for p in lesson.key_points], "", "## 练习"]
    for i, (ex, ans, run, g) in enumerate(zip(lesson.exercises, lesson.answers, lesson.runs, lesson.grades), 1):
        parts += [f"### {i}. {ex['question']}{'（对）' if g.get('pass') else '（错）'}"]
        if ex.get("code"):
            parts += ["```", ex["code"], "```"]
        parts += ["我的答案：", "```", (_code_of(ans) if ex.get("kind") == "write" else ans.strip())[:1200], "```"]
        if run:
            parts.append(f"运行：{run}")
        parts.append(f"老师：{g.get('comment', '')}")
        if not g.get("pass") and g.get("correct"):
            parts += ["正确答案：", "```", str(g["correct"]), "```"]
    parts += ["", f"总评：{lesson.summary}（{'掌握了' if lesson.mastered else '明天再练'}）"]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def draft_skill(teacher, spec: dict, skills_dir: Path) -> str:
    """A new ability as a skill pack; risk is always 'ask' (the owner approves each run)."""
    res = _json(teacher.complete(SKILL_SYSTEM, json.dumps(spec, ensure_ascii=False), json_mode=True, max_tokens=2500))
    name = re.sub(r"[^a-z0-9\-]", "-", str(res.get("name") or spec.get("name") or "").lower()).strip("-")[:40]
    script_name = str(res.get("script_name") or "")
    script, body = str(res.get("script") or ""), str(res.get("skill_md") or "")
    if not name or not body or not re.fullmatch(r"[\w\-]+\.(sh|py)", script_name):
        return ""
    if why := (check_script(script) or check_script(body)):
        LOG.warning("[school] skill draft %s rejected: %s", name, why)
        return ""
    d = skills_dir / name
    if d.exists():
        return ""  # never overwrite a skill (shipped or approved)
    (d / "scripts").mkdir(parents=True)
    desc = str(res.get("description") or spec.get("description") or name).replace("\n", " ")
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\nrisk: ask\ntags: {res.get('tags', '')}\n"
                                f"source: school (老师写的草稿，每次执行都要主人确认)\n---\n{body}\n", encoding="utf-8")
    (d / "scripts" / script_name).write_text(script, encoding="utf-8")
    return name


def school_day(*, teacher, store, state: SchoolState, local_answer: Callable[[str], str], log_training,
               identity: str, abilities: str, troubles: str, notes_dir: Path, skills_dir: Path,
               topic: str | None = None, idle: Callable[[], bool] = lambda: True,
               run: Callable[[str, str], tuple[bool, str]] = run_sandboxed) -> dict:
    day = time.strftime("%Y-%m-%d")
    review = self_review(teacher, identity=identity, abilities=abilities, troubles=troubles)
    report = {"day": day, "who_am_i": review.get("who_am_i", ""), "purpose": review.get("purpose", ""),
              "gaps": review.get("gaps") or [], "lessons": [], "skill": ""}
    plan = [(topic or state.next_topic(), "syllabus")]
    if review.get("lesson_topic") and not topic:
        plan.append((str(review["lesson_topic"]), "gap"))
    for t, kind in plan:
        if not idle():
            report["stopped"] = True
            break
        lesson = teach(teacher, t, kind, local_answer, run)
        path = remember(lesson, store=store, log_training=log_training, notes_dir=notes_dir, day=day)
        passed = sum(1 for g in lesson.grades if g.get("pass"))
        report["lessons"].append({"topic": t, "kind": kind, "title": lesson.title, "passed": passed,
                                  "total": len(lesson.exercises), "mastered": lesson.mastered, "notes": str(path)})
        if kind == "syllabus" and lesson.mastered and not topic:
            state.index += 1
        state.history.append({"day": day, "topic": t, "passed": passed, "total": len(lesson.exercises),
                              "mastered": lesson.mastered})
    spec = review.get("ability_skill")
    if isinstance(spec, dict) and spec.get("name"):
        try:
            report["skill"] = draft_skill(teacher, spec, skills_dir)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[school] skill draft failed: %s", exc)
    state.save()
    diary = notes_dir / f"{day}-自省.md"
    lines = [f"# {day} 自省", "", f"我是谁：{report['who_am_i']}", f"我要干什么：{report['purpose']}", "", "我缺什么："]
    lines += [f"- {g.get('gap', '')}（{g.get('why', '')}）" for g in report["gaps"] if isinstance(g, dict)]
    lines += ["", "今天上的课："] + [f"- {x['title']}：{x['passed']}/{x['total']}，{'掌握了' if x['mastered'] else '明天再练'}"
                                       for x in report["lessons"]]
    if report["skill"]:
        lines.append(f"\n新写的技能草稿：{report['skill']}（每次执行都会先问主人）")
    diary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def school_session(*, teacher, store, state: SchoolState, local_answer: Callable[[str], str], log_training,
                   identity: str, abilities: str, troubles: str, notes_dir: Path, skills_dir: Path,
                   run: Callable[[str, str], tuple[bool, str]] = run_sandboxed,
                   self_topics: Callable[[], list[tuple[str, str]]] | None = None, only_self: bool = False) -> dict:
    """One lesson ("一直上课": the timer calls this every half hour while the owner is away).
    The first lesson of a day starts with the self-review, which queues a lesson per gap; then lessons
    rotate syllabus -> 认识自己 (her own hardware and code, when self_topics is given) -> queued gap
    lessons; a gap lesson not mastered goes back in the queue, a self lesson is retried once."""
    day = time.strftime("%Y-%m-%d")
    out: dict = {"day": day, "review": False, "skill": ""}
    learning_dir = state.path.parent.parent
    if state.day != day:
        now.begin(learning_dir, "review", "每天的自省", "想想我是谁、缺什么")
        review = self_review(teacher, identity=identity, abilities=abilities, troubles=troubles)
        state.day, state.review = day, {k: review.get(k) for k in ("who_am_i", "purpose", "gaps")}
        topics = [str(review["lesson_topic"])] if review.get("lesson_topic") else []
        topics += [f"补短板：{g['gap']}" for g in review.get("gaps") or [] if isinstance(g, dict) and g.get("gap")]
        state.pending = list(dict.fromkeys(topics + state.pending))[:12]
        out["review"] = True
        spec = review.get("ability_skill")
        if isinstance(spec, dict) and spec.get("name"):
            try:
                out["skill"] = draft_skill(teacher, spec, skills_dir)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("[school] skill draft failed: %s", exc)
        diary = [f"# {day} 自省", "", f"我是谁：{review.get('who_am_i', '')}", f"我要干什么：{review.get('purpose', '')}",
                 "", "我缺什么："] + [f"- {g.get('gap', '')}（{g.get('why', '')}）" for g in review.get("gaps") or [] if isinstance(g, dict)]
        if out["skill"]:
            diary.append(f"\n新写的技能草稿：{out['skill']}（每次执行都会先问主人）")
        notes_dir.mkdir(parents=True, exist_ok=True)
        (notes_dir / f"{day}-自省.md").write_text("\n".join(diary) + "\n\n今天上的课：\n", encoding="utf-8")
    selfs = self_topics() if self_topics else []
    slot = state.sessions % 3 if selfs else (1 + state.sessions % 2) % 3    # no self track: syllabus / gap
    order = {0: ("syllabus", "self", "gap"), 1: ("self", "syllabus", "gap") if selfs else ("syllabus", "gap"),
             2: ("gap", "syllabus", "self")}[slot]
    if only_self:
        order = ("self",)
    picked = None
    for track in order:   # this slot's track first; one with nothing left to teach hands over to the next
        if track == "gap" and state.pending:
            picked = (state.pending.pop(0), "gap", "")
        elif track == "self" and selfs and (got := state.due_self(selfs, day)):
            picked = (got[0], "self", got[1])
        elif track == "syllabus" and (t := state.due_syllabus(day)):
            picked = (t, "syllabus", "")
        if picked:
            break
    if picked is None:   # everything mastered within REVIEW_DAYS: no lesson, no teacher call
        state.sessions += 1
        state.save()
        out.update({"nothing": True, "topic": "", "kind": "", "title": "", "passed": 0, "total": 0, "mastered": False})
        return out
    topic, kind, material = picked
    label = {"syllabus": "课本", "self": "认识自己", "gap": "补短板"}.get(kind, kind)
    now.begin(learning_dir, "lesson", topic, f"第 {state.sessions + 1} 节 · {label}")
    try:
        lesson = teach(teacher, topic, kind, local_answer, run, material=material)
    finally:
        now.end(learning_dir)
    path = remember(lesson, store=store, log_training=log_training, notes_dir=notes_dir, day=day)
    passed = sum(1 for g in lesson.grades if g.get("pass"))
    if lesson.mastered:
        state.mastered[topic] = day
    if kind == "syllabus" and lesson.mastered:
        state.index += 1
    if kind == "gap" and not lesson.mastered:
        state.pending.append(topic)  # try again later, from another angle
    if kind == "self":
        state.self_tries += 1
        if lesson.mastered or state.self_tries >= 2:   # move on after a second try: there are hundreds of pieces
            state.self_index, state.self_tries = state.self_index + 1, 0
    state.sessions += 1
    state.history.append({"day": day, "time": time.strftime("%H:%M"), "topic": topic, "kind": kind, "passed": passed,
                          "total": len(lesson.exercises), "mastered": lesson.mastered})
    state.save()
    with (notes_dir / f"{day}-自省.md").open("a", encoding="utf-8") as f:
        f.write(f"- {time.strftime('%H:%M')} {lesson.title}：{passed}/{len(lesson.exercises)}，{'掌握了' if lesson.mastered else '还要再练'}\n")
    out.update({"topic": topic, "kind": kind, "title": lesson.title, "passed": passed, "total": len(lesson.exercises),
                "mastered": lesson.mastered, "notes": str(path)})
    return out


# ---------------------------------------------------------------- wiring
def owner_brief(owner_dir: Path) -> str:
    """What she knows about the owner so far (learning.owner / learning.inbox), for the self-review."""
    def load(name, default):
        try:
            return json.loads((owner_dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default
    prof = load("profile.json", {}).get("profile", [])
    open_q = [q["q"] for q in load("questions.json", []) if not q.get("answer")]
    n_sched = len(load("schedule.json", []))
    parts = ["\n\n我对主人的了解：" + ("；".join(prof[:10]) if prof else "还很少"),
             f"主人的课表/日程：{'有 ' + str(n_sched) + ' 项' if n_sched else '还没有'}"]
    if open_q:
        parts.append("还没得到回答的问题：" + "；".join(open_q[:5]))
    return "\n".join(parts)


def describe_abilities(agent_root: Path, skill_store) -> str:
    from learning import skillpack
    packs = skillpack.load([agent_root / "skills", agent_root / "run" / "skills"])
    learned = skill_store.all(30) if skill_store else []
    lines = ["技能包：" + "；".join(f"{p.name}（{p.description}）" for p in packs)]
    lines.append("学会的做法：" + ("；".join(s.task for s in learned) or "无"))
    lines.append("工具：在板子上执行命令、SSH 到已配置的机器、写笔记、读网页（MCP）、问老师 DeepSeek、本地小模型 Qwen3.5-4B（上下文 4096）")
    return "\n".join(lines)


def recent_troubles(journal_dir: Path, store, days: int = 3) -> str:
    from datetime import date, timedelta
    from learning.journal import read_day
    out = []
    for d in range(days):
        for t in read_day(journal_dir, date.today() - timedelta(days=d)):
            if (t.get("who") == "agent" and t.get("reason") == "failed") or re.search(r"不会|没法|做不了|不支持|没听", t.get("a", "")):
                out.append(f"- 主人说「{t['q'][:50]}」，我答「{t['a'][:80]}」")
    out += [f"- 答错过：{q[:50]}" for q, _, _ in store.lessons(limit=5)]
    return "\n".join(out[-15:])


def main() -> None:
    from learning.reflect import build_runtime
    from learning.brain import Learner
    from learning.practice import IDLE_MIN, last_activity
    from learning.skills import SkillStore

    ap = argparse.ArgumentParser(description="星期一 daily school")
    ap.add_argument("--config", default="/userdata/agent/config/agent.yaml")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--topic", help="teach this topic instead of the syllabus (no gap lesson)")
    ap.add_argument("--full-day", action="store_true", help="self-review + two lessons in one go (the old daily mode)")
    ap.add_argument("--self", dest="only_self", action="store_true", help="give the next 认识自己 lesson now")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    rt = build_runtime(args.config)
    run_dir = rt.learning_dir.parent
    agent_root = run_dir.parent

    def idle() -> bool:
        return args.force or time.time() - last_activity(run_dir) > IDLE_MIN * 60

    if not idle():
        LOG.info("[school] owner active: skip this lesson (next timer slot tries again)")
        return
    learner = Learner(rt.store, rt.learning_dir / "lora_data.jsonl")
    skills = SkillStore(rt.learning_dir / "memory.db")
    from llm.client import llm_chat
    from orchestrator.config_loader import load_yaml
    sock = str(load_yaml(args.config).get("socket_path", "/tmp/r1-llm.sock"))

    def student(q: str) -> str:
        """The local model as a student: a short system prompt (the 768-token context is for the lesson)
        and room for code; nothing stripped except thinking tags (the voice cleanup would eat code)."""
        prompt = (f"<|im_start|>system\n{CODING_SYSTEM}<|im_end|>\n"
                  f"<|im_start|>user\n{q[:900]}<|im_end|>\n<|im_start|>assistant\n")
        text = str(llm_chat(prompt, sock_path=sock, max_new_tokens=STUDENT_MAX_TOKENS).get("text", ""))
        return re.sub(r"(?s)<think>.*?</think>", "", text).strip()

    def log_training(q: str, a: str) -> None:
        learner.log_training(q, a, system=CODING_SYSTEM)   # coding mode, not the voice persona

    state = SchoolState.load(rt.learning_dir / "school" / "state.json")
    from learning import project, selfknow, weakspots
    if state.day != time.strftime("%Y-%m-%d") or not (rt.learning_dir / "self" / "hardware.json").is_file():
        try:   # once a day: re-survey the board (read-only) and re-map her code
            LOG.info("[school] self survey: %s", selfknow.write_survey(rt.learning_dir, agent_root))
        except Exception as exc:  # noqa: BLE001 - class goes on without the self track
            LOG.warning("[school] self survey failed: %s", exc)
    common = dict(teacher=rt.teacher, store=rt.store, state=state, local_answer=student, log_training=log_training,
                  identity=rt.system + owner_brief(rt.learning_dir / "owner"), abilities=describe_abilities(agent_root, skills),
                  troubles="\n".join(filter(None, [recent_troubles(rt.journal_dir, rt.store), weakspots.summary(rt.learning_dir),
                                                   project.review_lines(rt.learning_dir)])),
                  notes_dir=rt.learning_dir / "school",
                  skills_dir=run_dir / "skills")
    if args.full_day or args.topic:
        report = school_day(**common, topic=args.topic, idle=idle)
        for x in report["lessons"]:
            LOG.info("[school] %s: %d/%d %s", x["title"], x["passed"], x["total"], "mastered" if x["mastered"] else "again")
        return
    r = school_session(**common, self_topics=lambda: selfknow.self_topics(agent_root, rt.learning_dir),
                       only_self=args.only_self)
    if r["review"]:
        LOG.info("[school] self-review: %s | gaps: %s", state.review.get("who_am_i"),
                 "；".join(g.get("gap", "") for g in state.review.get("gaps") or [] if isinstance(g, dict)))
    if r.get("nothing"):
        LOG.info("[school] nothing left to teach: every topic was mastered within %d days (no teacher call)", REVIEW_DAYS)
        return
    LOG.info("[school] lesson %d (%s) %s: %d/%d %s | next syllabus: %s | queued gap lessons: %d", state.sessions, r["kind"],
             r["title"], r["passed"], r["total"], "mastered" if r["mastered"] else "again later", state.next_topic(), len(state.pending))
    if r["skill"]:
        LOG.info("[school] drafted skill: %s", r["skill"])


if __name__ == "__main__":
    main()
