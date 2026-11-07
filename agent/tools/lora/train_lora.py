"""LoRA fine-tune of Qwen3.5-4B on 星期一's curated turns (runs on the GPU server, not the board).

    python train_lora.py --data train.jsonl --base models/Qwen3.5-4B --out out/lora_v1 [--epochs 3] [--smoke]

The prompt format is the exact ChatML the orchestrator sends to the board (no thinking block), and
the loss covers only the assistant reply + <|im_end|>. Only language-model layers get adapters; the
vision tower is untouched so the merged model exports with the stock rknn3 Qwen3_5 scripts.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)"


def chatml(messages: list[dict]) -> tuple[str, str]:
    """(prompt, reply) exactly as orchestrator/main.py builds them."""
    prompt = ""
    for m in messages[:-1]:
        prompt += f"<|im_start|>{m['role']}\n{m['content'].strip() if m['role'] == 'user' else m['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt, messages[-1]["content"].strip() + "<|im_end|>"


def encode(tok, rows: list[dict], max_len: int) -> list[tuple[list[int], list[int]]]:
    out = []
    for r in rows:
        prompt, reply = chatml(r["messages"])
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        a = tok(reply, add_special_tokens=False)["input_ids"]
        if len(a) > max_len // 2:
            a = a[: max_len // 2]
        p = p[-(max_len - len(a)):]          # keep the end of the prompt (the user turn)
        out.append((p + a, [-100] * len(p) + a))
    return out


@torch.no_grad()
def eval_loss(model, data) -> float:
    model.eval()
    tot, n = 0.0, 0
    for ids, lab in data:
        x = torch.tensor([ids], device="cuda")
        y = torch.tensor([lab], device="cuda")
        loss = model(input_ids=x, labels=y).loss
        k = sum(1 for t in lab if t != -100)
        tot += loss.item() * k
        n += k
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def sample(model, tok, rows, k=3, max_new=80) -> list[dict]:
    model.eval()
    res = []
    for r in rows[:k]:
        prompt, ref = chatml(r["messages"])
        x = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        y = model.generate(**x, max_new_tokens=max_new, do_sample=False,
                           eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
        res.append({"user": r["messages"][-2]["content"][:120], "ref": ref[:200],
                    "got": tok.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True)[:200]})
    model.train()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true", help="4 steps on 8 samples, no save")
    args = ap.parse_args()

    random.seed(0)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    random.shuffle(rows)
    n_eval = max(4, len(rows) // 10)
    ev_rows, tr_rows = rows[:n_eval], rows[n_eval:]
    if args.smoke:
        tr_rows, ev_rows = tr_rows[:8], ev_rows[:2]

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    tr, ev = encode(tok, tr_rows, args.max_len), encode(tok, ev_rows, args.max_len)
    print(f"train {len(tr)} eval {len(ev)} | max tokens {max(len(i) for i, _ in tr)}", flush=True)
    log = {"before": {"eval_loss": eval_loss(model, ev), "samples": sample(model, tok, ev_rows)}}
    print("eval loss before:", round(log["before"]["eval_loss"], 4), flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = 4 if args.smoke else math.ceil(len(tr) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    model.train()
    step, t0, run = 0, time.time(), []
    order = []
    while step < steps:
        if not order:
            order = list(range(len(tr)))
            random.shuffle(order)
        for _ in range(args.accum):
            if not order:
                break
            ids, lab = tr[order.pop()]
            loss = model(input_ids=torch.tensor([ids], device="cuda"), labels=torch.tensor([lab], device="cuda")).loss
            (loss / args.accum).backward()
            run.append(loss.item())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 5 == 0 or step == steps or args.smoke:
            print(f"step {step}/{steps} loss {sum(run) / len(run):.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
            run = []

    log["after"] = {"eval_loss": eval_loss(model, ev), "samples": sample(model, tok, ev_rows)}
    print("eval loss after:", round(log["after"]["eval_loss"], 4), flush=True)
    log["args"] = vars(args)
    if not args.smoke:
        out = Path(args.out)
        model.save_pretrained(out)
        (out / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False, indent=1)[:4000])
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
