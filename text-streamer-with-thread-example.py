import os
from threading import Thread

import torch
import transformers
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer
import time

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

prompt_template = \
    '''
    {question}
    '''


device = "auto"
trust_remote_code = False

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=trust_remote_code,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=trust_remote_code,
    config=model_config,
    quantization_config=bnb_config,
    device_map=device,
    torch_dtype=torch.bfloat16
)


def generate_text(prompt, llm_model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer)
    input_ids = inputs.input_ids.to(llm_model.device)
    attention_mask = inputs.attention_mask.to(llm_model.device)

    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        max_new_tokens=2048,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    generated_text = ""
    thread.start()
    for new_text in streamer:
        generated_text += new_text
        print (new_text, end='')

while True:
    user_question = input("Please enter your question: ")
    if user_question.lower() == 'exit':
        break
    prompt = prompt_template.format(question=user_question)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("start")
    generate_text(prompt, model, tokenizer)
    print("enx")

