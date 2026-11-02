import os
os.environ["HF_HOME"] = "E:/huggingface_cache"
import torch
import torch.nn as nn
import open_clip
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# stała liczba tokenów obrazu dla modelu ViT
NUM_PATCHES = 196

# konektor (tłumaczy wektory obrazu na język LLaMY)
class MLP(nn.Module):
    def __init__(self, in_features=768, out_features=4096):
        super(MLP, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, x):
        return self.proj(x)


class MedicalVLM(nn.Module):
    def __init__(self, hf_token, in_features=768, out_features=4096):
        super(MedicalVLM, self).__init__()

        # BiomedCLIP
        print("Ładowanie BiomedCLIP'a...")
        self.vision_model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        # zamrożenie wag modelu wizyjnego
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.vision_model.eval()

        # MLP
        print("Ładowanie MLP...")
        self.connector = MLP(in_features=in_features, out_features=out_features)

        # LLM
        llm_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        print(f"Ładowanie LLM ({llm_path})...")

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # padding z prawej, żeby token [BOS] był zawsze na samym początku
        self.tokenizer.padding_side = "right"

        # config kwantyzacji - do 4 bitów
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # config LLM'a
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            token=hf_token
        )

        # zamrożenie wag LLaMY przed dodaniem adapterów
        for param in self.llm.parameters():
            param.requires_grad = False

        # zrzucenie modeli na GPU
        self.vision_model.to(self.llm.device)
        self.connector.to(self.llm.device)

    # zawsze zamrażanie - protekcja
    def train(self, mode=True):
        super().train(mode)
        self.vision_model.eval()
        return self

    def _build_inputs(self, images, text_input_ids, attention_mask):
        # wyciągnięcie i normalizacja łatek obrazu
        with torch.no_grad():
            vf = self.vision_model.visual.trunk.forward_features(images)
            patch = vf[:, 1:, :]
            patch = patch / patch.norm(dim=-1, keepdim=True)

        # przepuszczenie przez konektor
        image_embeds = self.connector(patch.float()).to(self.llm.dtype)

        with torch.no_grad():
            text_embeds = self.llm.get_input_embeddings()(text_input_ids)

        # konstrukcja: [BOS][196 tokenów obrazu][reszta promptu]
        inputs_embeds = torch.cat(
            [text_embeds[:, :1, :], image_embeds, text_embeds[:, 1:, :]], dim=1
        )

        # wymuszenie gradientu dla LoRa
        if self.training and not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad_(True)

        # budowanie attention maski
        if attention_mask is not None:
            image_mask = torch.ones((attention_mask.shape[0], NUM_PATCHES), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat(
                [attention_mask[:, :1], image_mask, attention_mask[:, 1:]], dim=1
            )
        return inputs_embeds, attention_mask

    # przejście w przód - liczenie logits dla straty
    def forward(self, images, text_input_ids, attention_mask=None):
        inputs_embeds, attention_mask = self._build_inputs(images, text_input_ids, attention_mask)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

    # generowanie gotowego tekstu
    @torch.no_grad()
    def generate(self, images, text_input_ids, attention_mask=None, **kwargs):
        inputs_embeds, attention_mask = self._build_inputs(images, text_input_ids, attention_mask)
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs
        )