import os
os.environ["HF_HOME"] = "E:/huggingface_cache"
import torch
import torch.nn as nn
import open_clip
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

        # zamrożenie wag
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # MLP
        print("Inicjalizacja konektora MLP...")
        self.connector = MLP(in_features=in_features, out_features=out_features)

        # LLM
        llm_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        print(f"Ładowanie LLM ({llm_path})...")

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # kwantyzacja do 4 bitów
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        # ładowanie wag + config kwantyzacji
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            token=hf_token
        )

        # zamrożenie wag
        for param in self.llm.parameters():
            param.requires_grad = False

        # operacje na GPU
        self.vision_model.to(self.llm.device)
        self.connector.to(self.llm.device)

    def forward(self, images, text_input_ids, attention_mask=None):
        with torch.no_grad():
            # jednak zamiast 1 tokena -> 196 na ilośc łatek
            vision_features = self.vision_model.visual.trunk.forward_features(images)
            patch_features = vision_features[:, 1:, :]
            patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

        image_embeds = self.connector(patch_features).to(self.llm.dtype)

        with torch.no_grad():
            text_embeds = self.llm.get_input_embeddings()(text_input_ids)

        # łączenie tokenów łatek z promptem
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        if attention_mask is not None:
            image_mask = torch.ones((attention_mask.shape[0], 196), dtype=attention_mask.dtype,
                                    device=attention_mask.device)
            attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs

    def generate(self, images, text_input_ids, attention_mask=None, max_new_tokens=100):
        with torch.no_grad():
            vision_features = self.vision_model.visual.trunk.forward_features(images)
            patch_features = vision_features[:, 1:, :]
            patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

        image_embeds = self.connector(patch_features).to(self.llm.dtype)

        with torch.no_grad():
            text_embeds = self.llm.get_input_embeddings()(text_input_ids)

        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        if attention_mask is not None:
            image_mask = torch.ones((attention_mask.shape[0], 196), dtype=attention_mask.dtype,
                                    device=attention_mask.device)
            attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7
        )
        return generated_ids