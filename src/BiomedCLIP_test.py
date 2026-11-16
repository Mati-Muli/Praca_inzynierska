import torch
import numpy as np
import open_clip
import re
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataset import DatasetLoad

PATHOLOGIES = ['atelectasis', 'cardiomegaly', 'consolidation', 'opacity',
                      'edema', 'pleural effusion', 'granuloma']
REPORTS_CSV = "E:/Praca_inzynierska/data/indiana_reports.csv"
PROJECTIONS_CSV = "E:/Praca_inzynierska/data/indiana_projections.csv"
IMAGES_DIR = "E:/Praca_inzynierska/data/images/images_normalized"


class ZeroSHot(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_tensor, _ = self.base_dataset[idx]
        row = self.base_dataset.file.iloc[idx]
        combined_text = str(row.get('MeSH', '')).lower() + " " + str(row.get('findings', '')).lower()
        label = np.zeros(len(PATHOLOGIES), dtype=int)
        for i, pathology in enumerate(PATHOLOGIES):
            if pathology in combined_text:
                # szuaknie negacjiw raportach
                if not re.search(rf"(no|without|negative for|clear of|free of)[\s\w]{{0,30}}{pathology}",combined_text):
                    label[i] = 1

        return img_tensor, label


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    text_features_list = []
    with torch.no_grad():
        for path in PATHOLOGIES:
            tokens = tokenizer(["normal chest x-ray", f"chest x-ray showing {path}"]).to(device)
            feat = model.encode_text(tokens)
            text_features_list.append(feat / feat.norm(dim=-1, keepdim=True))

    # zbiór validacyjny
    dataset = DatasetLoad(REPORTS_CSV, PROJECTIONS_CSV, IMAGES_DIR, split='val', is_training=False)
    loader = DataLoader(ZeroSHot(dataset), batch_size=32, shuffle=False, num_workers=2)

    y_true, y_scores = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testowanie"):
            images = images.to(device)
            img_features = model.encode_image(images)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            batch_scores = []
            for i in range(len(PATHOLOGIES)):
                sim = (100.0 * img_features @ text_features_list[i].T).softmax(dim=-1)
                batch_scores.append(sim[:, 1].cpu().numpy())

            y_scores.append(np.stack(batch_scores, axis=1))
            y_true.append(labels.numpy())

    y_true = np.vstack(y_true)
    y_scores = np.vstack(y_scores)

    print("-" * 50)
    for i, pathology in enumerate(PATHOLOGIES):
        cases = int(np.sum(y_true[:, i]))
        auc = roc_auc_score(y_true[:, i], y_scores[:, i]) if cases > 0 else 0.0
        print(f"{pathology.upper():<16} | ROC AUC: {auc:.4f} | Prawdziwe przypadki: {cases}")
    print("-" * 50)

if __name__ == "__main__":
    main()