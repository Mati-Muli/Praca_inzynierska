import os
os.environ["HF_HOME"] = "E:/huggingface_cache"
import sys
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from bert_score import score
import sklearn.utils.sparsefuncs
import radeval.metrics._chexbert_base
_orig_count_nonzero = sklearn.utils.sparsefuncs.count_nonzero

def chexbert_patch(X, axis=None):
    if type(X).__name__ == 'matrix':
        X = np.asarray(X)
    if not hasattr(X, 'format'):
        return np.count_nonzero(X, axis=axis)
    return _orig_count_nonzero(X, axis=axis)


radeval.metrics._chexbert_base.count_nonzero = chexbert_patch
from radeval import RadEval
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TOKEN import HF_TOKEN
from src.dataset import DatasetLoad
from src.bridge import MedicalVLM

REPORTS_CSV = "E:/Praca_inzynierska/data/indiana_reports.csv"
PROJECTIONS_CSV = "E:/Praca_inzynierska/data/indiana_projections.csv"
IMAGES_DIR = "E:/Praca_inzynierska/data/images/images_normalized"

CKPT_DIR = "E:/Praca_inzynierska/checkpoints"
MLP_PATH = os.path.join(CKPT_DIR, "wagi_mlp.pt")
LORA_PATH = os.path.join(CKPT_DIR, "lora_adaptery")
OUTPUT_DIR = Path("E:/Praca_inzynierska/wyniki_test")

MAX_SAMPLES = None
NUM_BEAMS = 1
REPETITION_PENALTY = 1.0
NO_REPEAT_NGRAM = 3
MAX_NEW_TOKENS = 256
FILENAME = f"beams{NUM_BEAMS}_rp{REPETITION_PENALTY}_nr{NO_REPEAT_NGRAM}_VLM"

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "You are an expert radiologist. Analyze this chest X-ray and give detailed report."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def compute_metrics(refs: list, hyps: list) -> dict:
    all_results = {}
    metric_groups = [
        ("NLG (BLEU, ROUGE)", ["bleu", "rouge"]),
        ("Kliniczne (CheXbert)", ["f1chexbert"]),
    ]

    for name, metrics in metric_groups:
        print(f"\nObliczanie: {name} ...")
        try:
            try:
                evaluator = RadEval(metrics=metrics, detailed=True)
            except TypeError:
                evaluator = RadEval(metrics=metrics)

            all_results.update(evaluator(refs=refs, hyps=hyps))
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"Błąd podczas liczenia {name}: {err_msg}")
            all_results[f"{name}_error"] = err_msg

    print(f"\nObliczanie: Semantyczne (BERTScore) ...")
    try:
        P, R, F1 = score(hyps, refs, lang="en", verbose=True)
        all_results["bertscore_precision"] = P.mean().item()
        all_results["bertscore_recall"] = R.mean().item()
        all_results["bertscore_f1"] = F1.mean().item()

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print(f"Błąd podczas liczenia BERTScore: {err_msg}")
        all_results["bertscore_error"] = err_msg

    return all_results


def save_pretty_metrics_report(metrics: dict, config_name: str, num_samples: int, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f" RAPORT Z EWALUACJI MODELU".center(60) + "\n")
        f.write(f"Konfiguracja      : {config_name}\n")
        f.write(f"Liczba zdjęć      : {num_samples}\n")

        for key, value in metrics.items():
            if "error" in key:
                f.write(f"[!] BŁĄD w {key.replace('_error', '')}:\n    {value}\n\n")
                continue

            if isinstance(value, dict):
                f.write(f"[{key.upper()}]\n")
                for sub_k, sub_v in value.items():
                    if isinstance(sub_v, float):
                        f.write(f"  - {sub_k:<30}: {sub_v:.4f}\n")
                    else:
                        f.write(f"  - {sub_k:<30}: {sub_v}\n")
                f.write("\n")
            elif isinstance(value, float):
                f.write(f"{key:<34}: {value:.4f}\n")
            else:
                f.write(f"{key:<34}: {value}\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = MedicalVLM(hf_token=HF_TOKEN, atten="eager")
    dev = model.llm.device
    model.connector.load_state_dict(torch.load(MLP_PATH, map_location=dev, weights_only=True))
    model.llm.load_adapter(LORA_PATH)
    model.eval()

    terminators = [
        model.tokenizer.eos_token_id,
        model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "eos_token_id": terminators,
        "do_sample": False,
        "num_beams": NUM_BEAMS,
        "repetition_penalty": REPETITION_PENALTY,
    }
    if NO_REPEAT_NGRAM > 0:
        gen_kwargs["no_repeat_ngram_size"] = NO_REPEAT_NGRAM

    dataset = DatasetLoad(REPORTS_CSV, PROJECTIONS_CSV, IMAGES_DIR, split='test', is_training=False)
    num_samples = len(dataset) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(dataset))

    print(f"\nKonfiguracja generacji: {FILENAME}")
    print(f"{num_samples} próbek.")

    results = []

    for idx in tqdm(range(num_samples), desc="Generowanie raportów"):
        img_tensor, true_report = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(dev)

        row = dataset.file.iloc[idx]
        text_inputs = model.tokenizer(PROMPT_TEMPLATE, add_special_tokens=False, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                images=img_tensor,
                text_input_ids=text_inputs["input_ids"].to(dev),
                attention_mask=text_inputs["attention_mask"].to(dev),
                **gen_kwargs
            )

        generated = model.tokenizer.decode(output[0], skip_special_tokens=True).strip()

        results.append({
            "Indeks": idx,
            "Plik obrazu": row['filename'],
            "MeSH": row.get('MeSH', "Brak danych"),
            "Prawdziwy Raport": true_report,
            "Wygenerowany Raport": generated
        })

    csv_path = OUTPUT_DIR / f"wyniki_{num_samples}_probek_{FILENAME}.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nZapisano surowe wyniki do CSV: {csv_path}")

    predictions = [res["Wygenerowany Raport"] for res in results]
    references = [res["Prawdziwy Raport"] for res in results]

    n_empty = sum(1 for p in predictions if not p.strip())
    print(f"Uwaga: Puste wygenerowane raporty: {n_empty}/{len(predictions)}")

    print("\nObliczanie metryk...")
    metrics_results = compute_metrics(references, predictions)

    json_path = OUTPUT_DIR / f"metryki_{FILENAME}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_results, f, indent=4, default=str)

    txt_path = OUTPUT_DIR / f"podsumowanie_{FILENAME}.txt"
    save_pretty_metrics_report(metrics_results, FILENAME, num_samples, txt_path)

    print(f"\nZakończono")


if __name__ == "__main__":
    main()