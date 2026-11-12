import os
os.environ["HF_HOME"] = "/content/huggingface_cache"
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import get_cosine_schedule_with_warmup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TOKEN import HF_TOKEN
os.environ["HF_TOKEN"] = HF_TOKEN
from src.dataset import DatasetLoad
from src.bridge import MedicalVLM, NUM_PATCHES

REPORTS_CSV = "/content/data/indiana_reports.csv"
PROJECTIONS_CSV = "/content/data/indiana_projections.csv"
IMAGES_DIR = "/content/data/images/images_normalized"

CKPT_DIR = "/content/drive/MyDrive/Praca_inzynierska/checkpoints"
MLP_PATH = os.path.join(CKPT_DIR, "wagi_mlp.pt")
LORA_PATH = os.path.join(CKPT_DIR, "lora_adaptery")

PROMPT = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
          "You are an expert radiologist. Describe the findings in this chest X-ray."
          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
MAX_LEN = 384


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=['mlp', 'lora'], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"\n{'=' * 50}\nSTART TRENINGU | ETAP: {args.stage.upper()}\n{'=' * 50}\n")

    train_dataset = DatasetLoad(REPORTS_CSV, PROJECTIONS_CSV, IMAGES_DIR, split='train', is_training=True)
    val_dataset = DatasetLoad(REPORTS_CSV, PROJECTIONS_CSV, IMAGES_DIR, split='val', is_training=False)

    model = MedicalVLM(hf_token=HF_TOKEN)
    dev = model.llm.device

    # słownik słów do kar
    keywords = [
        "cardiomegaly", "enlarged", "effusion", "pneumothorax", "opacity", "opacities",
        "consolidation", "edema", "atelectasis", "infiltrate", "nodule", "mass", "scarring",
        "calcified", "granuloma", "atherosclerosis", "tortuous", "hernia", "hyperinflation",
        "blunting", "prominence", "congestion", "interstitial", "pneumonia", "deformity",
        "fracture", "surgical", "clips", "catheter", "pacemaker", "degenerative",
        "osteophytes", "sclerotic", "thickening", "pleural", "abnormality", "scar"
    ]
    Penalty = 10.0

    keyword_ids = set()
    for word in keywords:
        for variant in [word, " " + word, word.capitalize(), " " + word.capitalize(), word.upper(), " " + word.upper()]:
            tokens = model.tokenizer.encode(variant, add_special_tokens=False)
            keyword_ids.update(tokens)

    keyword_ids_tensor = torch.tensor(list(keyword_ids), device=dev)

    prompt_len = len(model.tokenizer(PROMPT, add_special_tokens=False)["input_ids"])

    if args.stage == 'lora':
        if os.path.exists(MLP_PATH):
            model.connector.load_state_dict(torch.load(MLP_PATH, map_location=dev, weights_only=True))
            print("Załadowano wagi MLP.")
        else:
            print(f"Brak pliku {MLP_PATH}.")
            sys.exit(1)

    groups = []

    # zawsze oblokowane mlp, uczenie nawet przy lora dla poprawy wag
    for p in model.connector.parameters():
        p.requires_grad = True

    # mniejszy lr mlp dla lory
    lr_mlp = 5e-4 if args.stage == 'mlp' else 2e-5
    groups.append({"params": list(model.connector.parameters()), "lr": lr_mlp})

    if args.stage == 'lora':
        try:
            model.llm = prepare_model_for_kbit_training(model.llm)
        except Exception:
            pass

        lora_config = LoraConfig(
            r=32, lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
        )
        model.llm = get_peft_model(model.llm, lora_config)
        model.llm.print_trainable_parameters()

        groups.append({"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": 2e-5})

    all_params = [p for g in groups for p in g["params"]]


    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch])
        full_texts = [PROMPT + str(b[1]).strip() + "<|eot_id|>" for b in batch]
        tok = model.tokenizer(full_texts, padding=True, truncation=True, max_length=MAX_LEN, add_special_tokens=False, return_tensors="pt")
        labels = tok["input_ids"].clone()
        labels[:, :prompt_len] = -100
        labels[tok["attention_mask"] == 0] = -100
        return images, tok["input_ids"], tok["attention_mask"], labels

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    def compute_loss(images, input_ids, attention_mask, labels):
        out = model(images, input_ids, attention_mask)
        ignore = torch.full((images.shape[0], NUM_PATCHES), -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([labels[:, :1], ignore, labels[:, 1:]], dim=1)

        logits = out.logits[:, :-1].float()
        targets = full_labels[:, 1:]

        ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100, reduction='none')

        targets_flat = targets.reshape(-1)
        is_medical_token = torch.isin(targets_flat, keyword_ids_tensor)

        weights = torch.ones_like(ce_loss)
        weights[is_medical_token] = Penalty

        weighted_loss = ce_loss * weights
        valid_tokens = (targets_flat != -100).sum()

        if valid_tokens > 0:
            return weighted_loss.sum() / valid_tokens
        else:
            return weighted_loss.sum() * 0.0

    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    total_steps = max(1, (len(train_loader) // args.grad_accum) * args.epochs)
    # 5% warmupu
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)

    best_val = float("inf")
    print("\nRozpoczęcie trenowania.")

    try:
        for epoch in range(args.epochs):
            print(f"\n--- EPOKA {epoch + 1}/{args.epochs} ---")
            model.train()
            total_train = 0.0
            optimizer.zero_grad()
            bar = tqdm(train_loader, desc=f"Trenowanie")

            for step, (images, input_ids, attention_mask, labels) in enumerate(bar):
                images, input_ids = images.to(dev), input_ids.to(dev)
                attention_mask, labels = attention_mask.to(dev), labels.to(dev)

                loss = compute_loss(images, input_ids, attention_mask, labels)
                (loss / args.grad_accum).backward()

                if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_train += loss.item()
                bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_train = total_train / len(train_loader)

            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for images, input_ids, attention_mask, labels in tqdm(val_loader, desc="Walidacja"):
                    images, input_ids = images.to(dev), input_ids.to(dev)
                    attention_mask, labels = attention_mask.to(dev), labels.to(dev)
                    total_val += compute_loss(images, input_ids, attention_mask, labels).item()

            avg_val = total_val / len(val_loader)

            print(f"Podsumowanie Epoki {epoch + 1}: Train Loss = {avg_train:.4f} | Val Loss = {avg_val:.4f}")
            if avg_val < best_val:
                best_val = avg_val
                print(f"Nowy najlepszy val loss...")
                if args.stage == 'mlp':
                    torch.save(model.connector.state_dict(), MLP_PATH)
                elif args.stage == 'lora':
                    model.llm.save_pretrained(LORA_PATH)
                    torch.save(model.connector.state_dict(), MLP_PATH)

    except KeyboardInterrupt:
        print("\nPrzerwano ręcznie.")


if __name__ == "__main__":
    train()