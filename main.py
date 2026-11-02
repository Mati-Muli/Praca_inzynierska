import os
os.environ["HF_HOME"] = "E:/huggingface_cache"
import sys
import torch
from peft import PeftModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TOKEN import HF_TOKEN
from src.dataset import DatasetLoad
from src.bridge import MedicalVLM

REPORTS_CSV = "E:/Praca_inzynierska/data/indiana_reports.csv"
PROJECTIONS_CSV = "E:/Praca_inzynierska/data/indiana_projections.csv"
IMAGES_DIR = "E:/Praca_inzynierska/data/images/images_normalized"

# te same ścieżki co w train.py
CKPT_DIR = "E:/Praca_inzynierska/checkpoints"
MLP_PATH = os.path.join(CKPT_DIR, "wagi_mlp.pt")
LORA_PATH = os.path.join(CKPT_DIR, "lora_adaptery")

PROMPT = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
          "You are an expert radiologist. Describe the findings in this chest X-ray."
          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def main(sample_idx=192):
    print("Ładowanie modelu...")
    model = MedicalVLM(hf_token=HF_TOKEN)
    dev = model.llm.device

    model.connector.load_state_dict(torch.load(MLP_PATH, map_location=dev, weights_only=True))
    model.llm = PeftModel.from_pretrained(model.llm, LORA_PATH)
    model.eval()

    dataset = DatasetLoad(REPORTS_CSV, PROJECTIONS_CSV, IMAGES_DIR, split='all', is_training=False)
    img_tensor, true_report = dataset[sample_idx]
    img_tensor = img_tensor.unsqueeze(0).to(dev)

    text_inputs = model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(dev)
    attention_mask = text_inputs["attention_mask"].to(dev)

    terminators = [
        model.tokenizer.eos_token_id,
        model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    print("\nGenerowanie raportu...")
    output_ids = model.generate(
        images=img_tensor,
        text_input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        repetition_penalty=1.0,
    )

    generated = model.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    print("\n" + "=" * 50 + "\nPRAWDZIWY RAPORT:\n" + true_report)
    print("\n" + "=" * 50 + "\nWYGENEROWANY RAPORT:\n" + generated)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()


    # TODO
    # undersampling - do trenowania 50/50 zdrowi-chorzy
    # mniej epok MLP więcej LORA