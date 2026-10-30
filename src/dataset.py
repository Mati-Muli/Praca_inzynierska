import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import albumentations as al
from albumentations.pytorch import ToTensorV2


def get_transforms(is_training=False):
    # resize i wyrównanie histogramu
    transforms_list = [
        al.Resize(224, 224),
        al.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    ]

    # obrocanie dla trenowania
    if is_training:
        transforms_list.append(
            al.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0)
        )

    # normalizacja
    transforms_list.extend([
        al.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
        ToTensorV2()
    ])

    return al.Compose(transforms_list)


class DatasetLoad(Dataset):
    def __init__(self, reports_csv, projections_csv, images_dir, split='train', is_training=False):
        self.images_dir = images_dir
        self.transform = get_transforms(is_training=is_training)

        reports = pd.read_csv(reports_csv)
        projections = pd.read_csv(projections_csv)
        df_all = pd.merge(reports, projections, on="uid", how="inner").dropna(subset=['findings'])

        df_train, df_temp = train_test_split(df_all, test_size=0.2, random_state=42)
        df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

        if split == 'train':
            df_selected = df_train
        elif split == 'val':
            df_selected = df_val
        elif split == 'test':
            df_selected = df_test
        elif split == 'all':
            df_selected = df_all
        else:
            raise ValueError("Bląd")

        self.file = df_selected.reset_index(drop=True)

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        row = self.file.iloc[idx]
        img_name = row['filename']
        img_path = os.path.join(self.images_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Nie znaleziono pliku obrazu: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        image_tensor = augmented['image']

        findings = str(row['findings']).strip()

        return image_tensor, findings