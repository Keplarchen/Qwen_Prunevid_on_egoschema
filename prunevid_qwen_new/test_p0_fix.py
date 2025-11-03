#!/usr/bin/env python3
"""Test if P0 fixes resolved the garbled output issue."""

import torch
from transformers import Qwen2_5_VLProcessor
from modeling_qwen2_5_vl_prunevid_dtd import Qwen2_5_VLForConditionalGeneration
from config import PruneVidConfig

# Test 1: Completely disable PruneVid (should work now after P0 fixes)
print("=" * 80)
print("Test 1: All stages disabled (post P0 fix)")
print("=" * 80)

config = PruneVidConfig(
    enable_stage1=False,
    enable_stage2=False,
    enable_cache_compression=False,
    verbose=True,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    prunevid_config=config,
)

processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Simple test without video (text only)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt").to(model.device)

print(f"\nModel prunevid_enabled: {model.model.prunevid_enabled}")
print(f"Model prunevid_config: {model.model.prunevid_config}")

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=50)

generated_text = processor.batch_decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(f"\nGenerated text: {generated_text}")

# Check if output is reasonable English
response_part = generated_text.split("assistant\n")[-1] if "assistant" in generated_text else generated_text
print(f"\nResponse only: {response_part}")

# Simple validation: check if contains mostly ASCII/English
ascii_ratio = sum(1 for c in response_part if ord(c) < 128) / max(len(response_part), 1)
print(f"ASCII ratio: {ascii_ratio:.2%}")

if ascii_ratio > 0.7 and len(response_part) > 0:
    print("\n✅ Test PASSED - Output appears to be valid English")
else:
    print("\n❌ Test FAILED - Output appears to be garbled")
