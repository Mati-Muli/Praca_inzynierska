import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import albumentations as al
from albumentations.pytorch import ToTensorV2


def get_transforms(is_training=False, use_clahe=True):
    transforms_list = [al.Resize(224, 224)]
    if use_clahe:
        transforms_list.append(al.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0))

    if is_training:
        transforms_list.append(
            al.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT)
        )

    # średnia/odchylenie CLIP - zgodne z BiomedCLIP
    transforms_list.extend([
        al.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
        ToTensorV2()
    ])
    return al.Compose(transforms_list)


class DatasetLoad(Dataset):
    def __init__(self, reports_csv, projections_csv, images_dir, split='train',
                 is_training=False, use_clahe=True, frontal_only=True):
        self.images_dir = images_dir
        self.transform = get_transforms(is_training=is_training, use_clahe=use_clahe)

        reports = pd.read_csv(reports_csv)
        projections = pd.read_csv(projections_csv)
        df_all = pd.merge(reports, projections, on="uid", how="inner").dropna(subset=['findings'])

        # same frontowe obrazy
        if frontal_only:
            df_all = df_all[df_all['projection'] == 'Frontal']

        # czyszczenie textu - model za bardzo uczył się słowa cenzury jako coś normalnego (XXXX)
        df_all = df_all.copy()
        df_all['findings'] = (df_all['findings'].astype(str)
                              .str.replace('XXXX', '', regex=False)
                              .str.replace(r'\s+', ' ', regex=True)
                              .str.strip())
        df_all = df_all[df_all['findings'].str.len() > 10]

        # podział po uid
        unique_uids = sorted(df_all['uid'].unique())
        train_uids, temp_uids = train_test_split(unique_uids, test_size=0.2, random_state=42)
        val_uids, test_uids = train_test_split(temp_uids, test_size=0.5, random_state=42)

        if split == 'train':
            df_selected = df_all[df_all['uid'].isin(train_uids)]
        elif split == 'val':
            df_selected = df_all[df_all['uid'].isin(val_uids)]
        elif split == 'test':
            df_selected = df_all[df_all['uid'].isin(test_uids)]
        elif split == 'all':
            df_selected = df_all
        else:
            raise ValueError(f"Nieznany split: {split}")

        self.file = df_selected.reset_index(drop=True)

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        row = self.file.iloc[idx]
        img_path = os.path.join(self.images_dir, row['filename'])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Nie znaleziono pliku obrazu: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_tensor = self.transform(image=image)['image']
        findings = str(row['findings']).strip()
        return image_tensor, findings