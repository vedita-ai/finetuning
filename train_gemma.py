import os
from unsloth import FastModel
import torch
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Load token from environment
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("❌ HF_TOKEN environment variable not set.")

# Load model
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-1b-it",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
)
print("✅ Model loaded.")

# Add PEFT (LoRA)
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)
print("🔧 PEFT (LoRA) applied.")

# Tokenizer template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
print("🧠 Tokenizer template set.")

# Load and format dataset
dataset = load_dataset("vaishnavi0901/unsloth-charak1", split="train", token=hf_token)
print("📦 Dataset loaded.")

dataset = standardize_data_formats(dataset)
print("🧹 Dataset standardized.")

def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return {"text": texts}

dataset = dataset.map(apply_chat_template, batched=False)
print("💬 Chat template applied to dataset.")

# Trainer config
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)
print("⚙️ Trainer configured.")

# Train
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)
print("🧪 Trainer prepared for response-only training.")

trainer_stats = trainer.train()
print("🏋️ Training complete.")

# Save and push
model.save_pretrained_merged("gemma-3-finetune1b", tokenizer)
print("💾 Model saved locally.")

model.push_to_hub_merged(
    "vaishnavi0901/gemma-3-finetune1b", tokenizer,
    token=hf_token
)
print("☁️ Model pushed to Hugging Face Hub.")
