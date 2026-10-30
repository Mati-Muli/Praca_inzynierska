import os
os.environ["HF_HOME"] = "E:/huggingface_cache"
import sys
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TOKEN import HF_TOKEN
from src.dataset import DatasetLoad
from src.bridge import MedicalVLM


def train():
    REPORTS_CSV = "E:/Praca_inzynierska/data/indiana_reports.csv"
    PROJECTIONS_CSV = "E:/Praca_inzynierska/data/indiana_projections.csv"
    IMAGES_DIR = "E:/Praca_inzynierska/data/images/images_normalized"

    EPOCHS = 3
    BATCH_SIZE = 2
    LR = 1e-4

    print(f"\n{'=' * 50}")
    print("URUCHOMIENIE UCZENIA | MLP")
    print(f"{'=' * 50}\n")

    train_dataset = DatasetLoad(
        reports_csv=REPORTS_CSV,
        projections_csv=PROJECTIONS_CSV,
        images_dir=IMAGES_DIR,
        split='train',
        is_training=True
    )

    val_dataset = DatasetLoad(
        reports_csv=REPORTS_CSV,
        projections_csv=PROJECTIONS_CSV,
        images_dir=IMAGES_DIR,
        split='val',
        is_training=False
    )

    print(f"Dane: Trening={len(train_dataset)}, Walidacja={len(val_dataset)}")

    model = MedicalVLM(hf_token=HF_TOKEN)

    try:
        model.connector.load_state_dict(torch.load("wagi_mlp.pt", weights_only=True))
        print("Znaleziono bazowe wagi MLP")
    except FileNotFoundError:
        print("Brak poprzednich wag.")

    for param in model.connector.parameters():
        param.requires_grad = True

    # optimizer Adam
    optimizer = torch.optim.AdamW(model.connector.parameters(), lr=LR)

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        texts = [item[1] for item in batch]

        # konstruckja promptu
        prompts = [
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are an expert radiologist. Describe the findings in this chest X-ray.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            for _ in texts
        ]

        # pełn tekst
        full_texts = [p + t + "<|eot_id|>" for p, t in zip(prompts, texts)]

        tokenized_full = model.tokenizer(full_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

        # obliczenie długości promptu do maskowania
        tokenized_prompts = model.tokenizer(prompts, add_special_tokens=False)
        prompt_lengths = [len(p) for p in tokenized_prompts["input_ids"]]

        return images, tokenized_full["input_ids"], tokenized_full["attention_mask"], prompt_lengths

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    loss_fn = CrossEntropyLoss(ignore_index=-100)

    # uczenie
    try:
        for epoch in range(EPOCHS):
            print(f"\n--- EPOKA {epoch + 1}/{EPOCHS} ---")

            model.train()
            total_train_loss = 0
            train_bar = tqdm(train_loader, desc="Uczenie")

            for step, (images, input_ids, attention_mask, prompt_lengths) in enumerate(train_bar):
                images = images.to(model.llm.device)
                input_ids = input_ids.to(model.llm.device)
                attention_mask = attention_mask.to(model.llm.device)
                optimizer.zero_grad()
                outputs = model(images, input_ids, attention_mask)
                logits = outputs.logits

                full_labels = torch.full((images.shape[0], 196 + input_ids.shape[1]), -100, dtype=torch.long, device=model.llm.device)
                full_labels[:, 196:] = input_ids.clone()

                # maskowanie
                full_labels[full_labels == model.tokenizer.pad_token_id] = -100

                # maskowanie
                for i, p_len in enumerate(prompt_lengths):
                    full_labels[i, 196: 196 + p_len] = -100

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = full_labels[:, 1:].contiguous()

                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / len(train_loader)

            # walidacja
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images, input_ids, attention_mask, prompt_lengths in val_loader:
                    images, input_ids, attention_mask = images.to(model.llm.device), input_ids.to(model.llm.device), attention_mask.to(model.llm.device)
                    outputs = model(images, input_ids, attention_mask)
                    logits = outputs.logits
                    full_labels = torch.full((images.shape[0], 196 + input_ids.shape[1]), -100, dtype=torch.long, device=model.llm.device)
                    full_labels[:, 196:] = input_ids.clone()
                    full_labels[full_labels == model.tokenizer.pad_token_id] = -100
                    for i, p_len in enumerate(prompt_lengths):
                        full_labels[i, 196: 196 + p_len] = -100

                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = full_labels[:, 1:].contiguous()

                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    total_val_loss += loss.item()
                    train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_val_loss = total_val_loss / len(val_loader)

            print(f"Epoka {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
            torch.save(model.connector.state_dict(), "wagi_mlp.pt")
            print("Zapisano wagi 'wagi_mlp.pt'.")

    except KeyboardInterrupt:
        print("\n\n[!] Przerwanie ręcznie.")
        torch.save(model.connector.state_dict(), "wagi_mlp.pt")
        print("Zapisano wagi 'wagi_mlp.pt'.")


if __name__ == "__main__":
    train()