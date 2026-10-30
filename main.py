import os
os.environ["HF_HOME"] = "E:/huggingface_cache"
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TOKEN import HF_TOKEN
from src.dataset import DatasetLoad
from src.bridge import MedicalVLM


def main():
    REPORTS_CSV = "E:/Praca_inzynierska/data/indiana_reports.csv"
    PROJECTIONS_CSV = "E:/Praca_inzynierska/data/indiana_projections.csv"
    IMAGES_DIR = "E:/Praca_inzynierska/data/images/images_normalized"

    print("Ładowanie modelu...")
    model = MedicalVLM(hf_token=HF_TOKEN)

    print("Wczytywanie danych z wagi_mlp.pt...")
    model.connector.load_state_dict(torch.load("src/wagi_mlp.pt", weights_only=True))
    model.eval()

    dataset = DatasetLoad(
        reports_csv=REPORTS_CSV,
        projections_csv=PROJECTIONS_CSV,
        images_dir=IMAGES_DIR,
        split='test',
        is_training=False
    )

    img_tensor, prawdziwy_raport = dataset[0]
    img_tensor = img_tensor.unsqueeze(0).to(model.llm.device)

    prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are an expert radiologist. Describe the findings in this chest X-ray.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    text_inputs = model.tokenizer(prompt, return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(model.llm.device)
    attention_mask = text_inputs["attention_mask"].to(model.llm.device)

    print("\nGenerowanie raportu...")
    with torch.no_grad():
        output_ids = model.generate(
            images=img_tensor,
            text_input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100
        )

    wygenerowany_tekst = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n" + "="*50)
    print("PRAWDZIWY RAPORT:")
    print(prawdziwy_raport)
    print("\n" + "="*50)
    print("WYGENEROWANY RAPORT:")
    print(wygenerowany_tekst)
    print("="*50 + "\n")


if __name__ == "__main__":
    main()