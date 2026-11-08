import os
os.environ["HF_HOME"] = "E:/huggingface_cache"
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TOKEN import HF_TOKEN
from src.dataset import DatasetLoad
from src.bridge import MedicalVLM
from src.visual import gen_heatmap

REPORTS_CSV = "E:/Praca_inzynierska/data/indiana_reports.csv"
PROJECTIONS_CSV = "E:/Praca_inzynierska/data/indiana_projections.csv"
IMAGES_DIR = "E:/Praca_inzynierska/data/images/images_normalized"
CKPT_DIR = "E:/Praca_inzynierska/checkpoints"
MLP_PATH = os.path.join(CKPT_DIR, "wagi_mlp.pt")
LORA_PATH = os.path.join(CKPT_DIR, "lora_adaptery")

OUTPUT_DIR = "E:/Praca_inzynierska/wyniki_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
          "You are an expert radiologist. Analyze this chest X-ray and give detailed raport."
          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def main():
    model = MedicalVLM(hf_token=HF_TOKEN, atten="eager")
    dev = model.llm.device
    model.connector.load_state_dict(torch.load(MLP_PATH, map_location=dev, weights_only=True))
    model.llm.load_adapter(LORA_PATH)
    model.eval()
    dataset = DatasetLoad(REPORTS_CSV, PROJECTIONS_CSV, IMAGES_DIR, split='test', is_training=False)

    terminators = [
        model.tokenizer.eos_token_id,
        model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    results = []
    num_samples = min(40, len(dataset))
    print(f"\nGenerowanie {num_samples} raportów...")

    for idx in tqdm(range(num_samples)):
        img_tensor, true_report = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(dev)

        row = dataset.file.iloc[idx]
        img_filename = row['filename']
        img_path = os.path.join(IMAGES_DIR, img_filename)

        text_inputs = model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")
        input_ids = text_inputs["input_ids"].to(dev)
        attention_mask = text_inputs["attention_mask"].to(dev)

        with torch.no_grad():
            output = model.generate(
                images=img_tensor,
                text_input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=False,
                repetition_penalty=1.1,
                output_attentions=True,
                return_dict_in_generate=True
            )

        output_ids = output.sequences[0]
        generated = model.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        heatmap_save_path = os.path.join(OUTPUT_DIR, f"heatmap_{idx:02d}_{img_filename}")
        fig = gen_heatmap(output.attentions, img_path)
        fig.savefig(heatmap_save_path)
        plt.close(fig)

        results.append({
            "Indeks": idx,
            "Plik obrazu": img_filename,
            "MeSH": row['MeSH'] if 'MeSH' in row else "Brak danych",
            "Prawdziwy Raport": true_report,
            "Wygenerowany Raport": generated
        })

    csv_path = os.path.join(OUTPUT_DIR, f"wyniki_{num_samples}_probek.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nZakończono")

if __name__ == "__main__":
    main()