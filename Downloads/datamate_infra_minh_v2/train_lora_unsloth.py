import argparse, json, torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

HOLDOUT = {"biz_105", "biz_118", "biz_124"}

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/train_generated_thinh.jsonl")
p.add_argument("--model", default="Qwen/Qwen3.5-9B")
p.add_argument("--out", default="qwen3.5-9b-datamate-lora")
p.add_argument("--max_len", type=int, default=16384)
p.add_argument("--epochs", type=float, default=2.0)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--r", type=int, default=16)
p.add_argument("--thinking", action="store_true")  # phai khop luc infer
p.add_argument("--all_linear", action="store_true")
p.add_argument("--no_merge", action="store_true")
a = p.parse_args()

model, tok = FastLanguageModel.from_pretrained(
    model_name=a.model, max_seq_length=a.max_len, dtype=torch.bfloat16, load_in_4bit=False)

print("Linear:", sorted({n.split(".")[-1] for n, m in model.named_modules()
                         if isinstance(m, torch.nn.Linear) and "visual" not in n}))

model = FastLanguageModel.get_peft_model(
    model, r=a.r, lora_alpha=a.r, lora_dropout=0, bias="none",
    target_modules="all-linear" if a.all_linear else
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=3407)

rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
keep = [r for r in rows if r.get("biz_id") not in HOLDOUT]
print(f"{len(keep)}/{len(rows)} rows (bo {len(rows) - len(keep)} dong holdout)")
if rows and "biz_id" not in rows[0]:
    print("WARNING: khong co truong biz_id, khong loc duoc holdout")

ds = Dataset.from_list([{"text": tok.apply_chat_template(
    r["messages"], tokenize=False, enable_thinking=a.thinking)} for r in keep])
print(ds[0]["text"][:1500])

tr = SFTTrainer(
    model=model, processing_class=tok, train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text", max_length=a.max_len, dataset_num_proc=4, packing=False,
        per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_ratio=0.03,
        num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit", weight_decay=0.01, logging_steps=5, seed=3407,
        save_strategy="epoch", output_dir="outputs", report_to="none"))
tr = train_on_responses_only(tr, instruction_part="<|im_start|>user\n",
                             response_part="<|im_start|>assistant\n")

if __name__ == "__main__":
    tr.train()
    model.save_pretrained(a.out + "-adapter")
    tok.save_pretrained(a.out + "-adapter")
    if not a.no_merge:
        model.save_pretrained_merged(a.out, tok, save_method="merged_16bit")
