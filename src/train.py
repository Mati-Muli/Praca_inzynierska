import os
from peft import PeftModel
os.environ["HF_HOME"] = "E:/huggingface_cache"
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import get_cosine_schedule_with_warmup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TOKEN import HF_TOKEN
from src.dataset import DatasetLoad
from src.bridge import MedicalVLM, NUM_PATCHES

REPORTS_CSV = "E:/Praca_inzynierska/data/indiana_reports.csv"
PROJECTIONS_CSV = "E:/Praca_inzynierska/data/indiana_projections.csv"
IMAGES_DIR = "E:/Praca_inzynierska/data/images/images_normalized"

CKPT_DIR = "E:/Praca_inzynierska/checkpoints"
MLP_PATH = os.path.join(CKPT_DIR, "wagi_mlp.pt")
LORA_PATH = os.path.join(CKPT_DIR, "lora_adaptery")

PROMPT = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
          "You are an expert radiologist. Analyze this chest X-ray and give detailed raport."
          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
MAX_LEN = 384


def train():
    parser = argparse.ArgumentParser(description="Trening MedicalVLM (MLP / LoRA / Both)")
    parser.add_argument("--stage", type=str, choices=['mlp', 'lora', 'both'], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--resume_mlp", action="store_true")
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"\n{'=' * 50}\nURUCHOMIENIE UCZENIA | TRYB: {args.stage.upper()}\n{'=' * 50}\n")

    train_dataset = DatasetLoad(REPORTS_CSV, PROJECTIONS_CSV, IMAGES_DIR, split='train', is_training=True)
    val_dataset = DatasetLoad(REPORTS_CSV, PROJECTIONS_CSV, IMAGES_DIR, split='val', is_training=False)
    print(f"Dane po balansie: Trening={len(train_dataset)}, Walidacja={len(val_dataset)}")

    model = MedicalVLM(hf_token=HF_TOKEN)
    dev = model.llm.device

    vocab_size = model.llm.config.vocab_size
    loss_weights = torch.ones(vocab_size, device=dev)

    # słownik słów do kar
    keywords = [
        "cardiomegaly", "enlarged", "effusion", "effusions", "pneumothorax",
        "pneumothoraces", "opacity", "opacities", "consolidation", "consolidations",
        "edema", "atelectasis", "infiltrate", "infiltrates", "nodule", "nodules",
        "mass", "masses", "scarring", "cicatrix", "calcified", "granuloma",
        "granulomas", "calcinosis", "atherosclerosis", "tortuous", "tortuosity",
        "hernia", "hiatal", "hypoinflation", "hyperinflation", "hyperdistention",
        "blunting", "blunted", "prominence", "prominent", "congestion",
        "interstitial", "disease", "pneumonia", "deformity", "fracture",
        "surgical", "clips", "hardware", "device", "catheter", "pacemaker",
        "degenerative", "osteophytes", "sclerotic", "thickening", "pleural",
        "airspace", "abnormality", "abnormalities", "scar", "scars"
    ]

    Penalty = 10.0

    for word in keywords:
        for variant in [word, " " + word, word.capitalize(), " " + word.capitalize(), word.upper(), " " + word.upper()]:
            tokens = model.tokenizer.encode(variant, add_special_tokens=False)
            for t_id in tokens:
                loss_weights[t_id] = Penalty
    print(f"Zainicjalizowano wagi straty dla {len(keywords)} słów kluczowych (mnożnik x{Penalty}).")

    prompt_len = len(model.tokenizer(PROMPT, add_special_tokens=False)["input_ids"])

    lens = []
    for t in train_dataset.file["findings"]:
        text_str = str(t).strip()
        tokens = model.tokenizer(PROMPT + text_str + "<|eot_id|>", add_special_tokens=False)["input_ids"]
        lens.append(len(tokens))
    too_long = sum(l > MAX_LEN for l in lens)
    print(f"Długość sekwencji: max={max(lens)}, średnia={sum(lens) / max(1, len(lens)):.0f}, "
          f"ponad {MAX_LEN} tokenów: {too_long}")
    if too_long > 0:
        print(f"Uwaga: {too_long} raportów zostanie przyciętych do {MAX_LEN} tokenów.")

    # ładowanie istniejących wag
    load_mlp = args.stage in ['lora', 'both'] or args.resume_mlp
    if os.path.exists(MLP_PATH) and load_mlp:
        model.connector.load_state_dict(torch.load(MLP_PATH, map_location=dev, weights_only=True))
        print("\nZaładowano wagi MLP.")
    elif args.stage == 'lora':
        print(f"\nBrak '{MLP_PATH}'. Najpierw wytrenuj MLP!")
        sys.exit(1)
    else:
        print("\nStart MLP od zera.")

    # zamrażanie/odmrażanie - wybór uczenia
    groups = []
    if args.stage in ['mlp', 'both']:
        for p in model.connector.parameters():
            p.requires_grad = True
        groups.append({"params": list(model.connector.parameters()), "lr": 5e-4})
    else:
        for p in model.connector.parameters():
            p.requires_grad = False

    if args.stage in ['lora', 'both']:
        model.llm = prepare_model_for_kbit_training(model.llm)
        lora_config = LoraConfig(
            r=32, lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM
        )

        if os.path.exists(os.path.join(LORA_PATH, "adapter_model.safetensors")):
            print(f"\nWznawianie treningu LoRA z {LORA_PATH}...")
            model.llm = PeftModel.from_pretrained(model.llm, LORA_PATH, is_trainable=True)
        else:
            print("\nTworzenie nowych adapterów LoRA od zera.")
            model.llm = get_peft_model(model.llm, lora_config)

        model.llm.print_trainable_parameters()
        groups.append({"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": 2e-5})
    else:
        print("LLM zamrożony.")

    all_params = [p for g in groups for p in g["params"]]

    # tworzenie batchy
    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch])
        full_texts = [PROMPT + str(b[1]) + "<|eot_id|>" for b in batch]
        tok = model.tokenizer(full_texts, padding=True, truncation=True, max_length=MAX_LEN, add_special_tokens=False,
                              return_tensors="pt")
        labels = tok["input_ids"].clone()
        labels[:, :prompt_len] = -100  # prompt nie wchodzi do loss
        labels[tok["attention_mask"] == 0] = -100  # padding
        return images, tok["input_ids"], tok["attention_mask"], labels

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Obliczanie straty z wykorzystaniem loss_weights
    def compute_loss(images, input_ids, attention_mask, labels):
        out = model(images, input_ids, attention_mask)
        ignore = torch.full((images.shape[0], NUM_PATCHES), -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([labels[:, :1], ignore, labels[:, 1:]], dim=1)
        logits = out.logits[:, :-1].float()
        targets = full_labels[:, 1:]

        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            weight=loss_weights
        )

    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)

    total_steps = max(1, (len(train_loader) // args.grad_accum) * args.epochs)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.03 * total_steps),
        num_training_steps=total_steps
    )

    # Zapis wag
    def save_models():
        if args.stage in ['mlp', 'both']:
            torch.save(model.connector.state_dict(), MLP_PATH)
            print(f"Zapisano konektor {MLP_PATH}")
        if args.stage in ['lora', 'both']:
            model.llm.save_pretrained(LORA_PATH)
            print(f"Zapisano adaptery {LORA_PATH}")

    # Główna pętla uczenia
    best_val = float("inf")
    try:
        for epoch in range(args.epochs):
            print(f"\n--- EPOKA {epoch + 1}/{args.epochs} ---")
            model.train()
            total_train = 0.0
            optimizer.zero_grad()
            bar = tqdm(train_loader, desc=f"Uczenie ({args.stage})")

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
                for images, input_ids, attention_mask, labels in val_loader:
                    images, input_ids = images.to(dev), input_ids.to(dev)
                    attention_mask, labels = attention_mask.to(dev), labels.to(dev)
                    total_val += compute_loss(images, input_ids, attention_mask, labels).item()
            avg_val = total_val / len(val_loader)

            print(f"Epoka {epoch + 1}: Train Loss = {avg_train:.4f} | Val Loss = {avg_val:.4f}")
            if avg_val < best_val:
                best_val = avg_val
                print(f"Best val loss: {best_val:.4f}")
                save_models()

    except KeyboardInterrupt:
        print("\nPrzerwano ręcznie.")


if __name__ == "__main__":
    train()