# ==============================================================================
# DATAMATE INFRA - LUONG B (MINH)
# LoRA SFT bang Unsloth cho Qwen3.5-9B (BF16, khong 4-bit)
#
# Dung:
#   python train_lora_unsloth.py --data data/train_generated_thinh.jsonl --epochs 2
#
# Dong data: {"messages":[{"role":"system",..},{"role":"user",..},{"role":"assistant",..}],
#             "biz_id": "biz_105"   # tuy chon, dung de loai 3 tiem giu rieng
#            }
# ==============================================================================
import argparse
import json

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

HOLDOUT = {"biz_105", "biz_118", "biz_124"}  # KHONG duoc dua vao train

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/train_generated_thinh.jsonl")
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--out", default="qwen3.5-9b-datamate-lora")
ap.add_argument("--max_seq_len", type=int, default=16384)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--thinking", action="store_true",
                help="Bat khi agent luc suy luan CO thinking. Phai khop luc infer.")
ap.add_argument("--all_linear", action="store_true",
                help="LoRA len moi lop Linear (dung neu Qwen3.5 co lop attention tuyen tinh ten khac)")
ap.add_argument("--no_merge", action="store_true")
args = ap.parse_args()

# 1. Model (BF16 LoRA, ~22GB VRAM)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    max_seq_length=args.max_seq_len,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# In ten cac lop Linear de kiem tra target_modules co phu het kien truc hybrid khong.
leaf = sorted({n.split(".")[-1] for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and "visual" not in n})
print("Linear leaf modules:", leaf)

targets = "all-linear" if args.all_linear else [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

model = FastLanguageModel.get_peft_model(
    model,
    r=args.rank,
    target_modules=targets,
    lora_alpha=args.rank,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 2. Data: doc jsonl, loai 3 tiem giu rieng, ap chat template -> cot "text"
rows, dropped = [], 0
with open(args.data, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("biz_id") in HOLDOUT:
            dropped += 1
            continue
        rows.append(r)
print(f"Loaded {len(rows)} rows, dropped {dropped} holdout rows")
if dropped == 0 and rows and "biz_id" not in rows[0]:
    print("WARNING: data khong co truong biz_id, khong loc duoc tiem giu rieng. "
          "Hay dam bao Thinh da bo biz_105/118/124.")


def to_text(r):
    return tokenizer.apply_chat_template(
        r["messages"], tokenize=False, add_generation_prompt=False,
        enable_thinking=args.thinking)


dataset = Dataset.from_list([{"text": to_text(r)} for r in rows])
print("Sample:\n", dataset[0]["text"][:1500])

# 3. Train
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_length=args.max_seq_len,
        dataset_num_proc=4,
        packing=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="epoch",
        output_dir="outputs",
        report_to="none",
    ),
)

# Chi tinh loss tren phan tra loi cua assistant (khong hoc lai prompt/system).
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

if __name__ == "__main__":
    print("Starting LoRA fine-tuning...")
    trainer.train()
    model.save_pretrained(args.out + "-adapter")      # adapter nho, dung cho vLLM --enable-lora
    tokenizer.save_pretrained(args.out + "-adapter")
    if not args.no_merge:
        model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
    print("Done.")
