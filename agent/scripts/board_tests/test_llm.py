import sys
import os
sys.path.insert(0, '/userdata/agent')

from llm.client import llm_chat_stream

def ask(q):
    prompt = (
        "<|im_start|>system\n"
        "你是 youyeetoo R1 智能语音助手小揽。\n"
        "你说话风格自然亲切、口语化、有问必答。\n"
        "【核心要求】：\n"
        "1. 必须使用纯中文口语回答，严禁输出任何英文字母或英文单词，遇到专业术语请使用中文全称。\n"
        "2. 严格禁止使用任何 Markdown 排版、列表符号或冒号。\n"
        "3. 遇到复杂问题时，请遵循金字塔法则：先用一句话给出核心结论，再简要解释原理，整体控制在两到三句话以内（60字以内），适合简短语音交流。\n"
        "4. 语句表达必须像真人面对面交谈一样自然流畅、结构完整、生动连贯。\n"
        "5. 严禁反问用户或者复读用户的问题，必须直接给出明确有用的事实回答。\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{q}\n<|im_end|>\n<|im_start|>assistant\n"
    )
    tokens = []
    res = llm_chat_stream(prompt, lambda tok: tokens.append(tok), sock_path='/tmp/r1-llm.sock', max_new_tokens=64)
    print(f"Q: {q} -> A: {repr(res['text'])}")

ask("中国的首都是哪里")
ask("法国的首都是哪里")
ask("首都是哪里")
ask("请问首都")
