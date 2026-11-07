import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def gen_heatmap(attentions, img_path):
    print("Przetwarzanie wag atencji...")

    all_token_attns = []

    for step_attns in attentions:
        # wyciąganie atencji z ostatniej warstwy llamy
        last_layer_attn = step_attns[-1][0]
        current_token_attn = last_layer_attn[:, -1, :]
        head_mean = current_token_attn.mean(dim=0)
        patch_attn = head_mean[2:198]
        all_token_attns.append(patch_attn)

    # sprawdzanie czy nie wystąpił attention sink
    avg_patch_attn = torch.stack(all_token_attns).mean(dim=0)
    print(avg_patch_attn.topk(10).indices.tolist())

    heatmap_14x14 = avg_patch_attn.reshape(14, 14).cpu().float().numpy()

    # normalizacja kolorów
    heatmap_14x14 = (heatmap_14x14 - heatmap_14x14.min()) / (heatmap_14x14.max() - heatmap_14x14.min() + 1e-8)

    orig_img = cv2.imread(img_path)
    if orig_img is None:
        raise FileNotFoundError(f"Nie znaleziono obrazu: {img_path}")
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (224, 224))

    # tworzenie maski
    heatmap_resized = cv2.resize(heatmap_14x14, (224, 224), interpolation=cv2.INTER_CUBIC)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(orig_img)
    axes[0].set_title("Oryginalne zdjęcie RTG", fontweight='bold')
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Mapa Atencji (Wyjaśnialność LLM)", fontweight='bold')
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()