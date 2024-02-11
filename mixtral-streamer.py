import os
from threading import Thread

import torch
import transformers
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, StoppingCriteria, \
    BitsAndBytesConfig
import time
from accelerate import Accelerator

model_id = "cognitivecomputations/dolphin-2.5-mixtral-8x7b"

PHI_PROMPT = \
    '''
{query}
Answer:
'''

accelerator = Accelerator()
device = "auto"
trust_remote_code = False
cache_dir = "/media/arko/work/.hfcache"

dm = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0,
      'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0,
      'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0,
      'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0,
      'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0,
      'model.layers.14': 0, 'model.layers.15': 0, 'model.layers.16': 1,
      'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1,
      'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 0,
      'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1,
      'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1,
      'model.layers.29': 1, 'model.layers.30': 1, 'model.layers.31': 1,
      'model.norm': 1, 'lm_head': 1}

compute_dtype = getattr(torch, "float16") # or bfloat16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=trust_remote_code,
    cache_dir=cache_dir
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=trust_remote_code,
    config=model_config,
    quantization_config=bnb_config,
    device_map=dm,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir
)




def generate_text(prompt, llm_model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextStreamer(tokenizer)
    input_ids = inputs.input_ids.to(llm_model.device)
    attention_mask = inputs.attention_mask.to(llm_model.device)

    start_time = time.time()
    outputs = llm_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=2048,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )
    end_time = time.time()
    # Check the type and structure of outputs
    if not isinstance(outputs, torch.Tensor) or len(outputs.shape) != 2:
        print("Unexpected output format from the model.")
        return None

    duration = end_time - start_time
    num_tokens = outputs.shape[
        1]  # Assuming outputs is a 2D tensor with shape [num_sequences, num_tokens]
    tokens_per_second = num_tokens / duration
    return tokens_per_second


while True:
    user_question = input("Please enter your question: ")
    if user_question.lower() == 'exit':
        break
    prompt = PHI_PROMPT.format(query=user_question)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, cache_dir=cache_dir)
    tps = generate_text(prompt, model, tokenizer)
    print(f"Token generation speed: {tps} tokens/second")
