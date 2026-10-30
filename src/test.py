import open_clip

_, _, preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

for transform in preprocess.transforms:
    if hasattr(transform, 'mean'):
        print(f"Mean: {transform.mean}")
        print(f"Std:  {transform.std}")