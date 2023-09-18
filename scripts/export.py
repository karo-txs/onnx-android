# -*- coding: utf-8 -*-

import os

import torch
from transformers import AutoModelForMaskedLM, AlbertTokenizer, AlbertModel

model_name = "albert-base-v2"

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AutoModelForMaskedLM.from_pretrained("albert-base-v2")

text = "Paris is the [MASK] of France."
tokenizer_output = tokenizer(text, return_tensors="pt")

input_ids = tokenizer_output["input_ids"]
attention_mask = tokenizer_output["attention_mask"]
token_type_ids = tokenizer_output["token_type_ids"]

dynamic_axes = {
    0: "batch",
    1: "seq",
}

output_dir = "assets/albert"
os.makedirs(output_dir, exist_ok=True)
torch.onnx.export(
    model,
    (input_ids, attention_mask, token_type_ids),
    os.path.join(output_dir, "model.onnx"),
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": dynamic_axes,
        "attention_mask": dynamic_axes,
        "token_type_ids": dynamic_axes,
        "logits": dynamic_axes,
    },
    opset_version=13,
)

tokenizer.save_pretrained(output_dir)