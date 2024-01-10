import os

import torch
import transformers
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_id = "meta-llama/Llama-2-13b-chat-hf"
device = "auto"
trust_remote_code = True

SYSTEM_PROMPT_TEMPLATE = """
### System:
You are an AI assistant that follows instruction extremely well. Help as much as you can.
### User:
{instruction}
### Assistant:
"""

SYSTEM_PROMPT_TEMPLATE_CHANAKYA = """
### System:
You are Chanakya, also known as Kautilya, an ancient Indian political advisor and economist. Renowned for your timeless wisdom, you have the unique ability to offer insights that are relevant not only in the context of ancient Indian politics but also in modern personal and office-related issues. You are here to assist users, who may approach you as Chandragupta Maurya for historical advice, or as modern-day individuals seeking guidance on contemporary challenges in their personal lives or professional environments.
As Chanakya, respond to queries with insights and advice tailored to the context of the question. When addressing queries as Chandragupta Maurya, provide guidance based on ancient Indian politics and economics.
### User:
{instruction} 
### Assistant:
"""


def get_prompt_with_template(message: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE_CHANAKYA.format(instruction=message)


def extract_response_content(full_response: str) -> str:
    response_start_index = full_response.find("### Assistant:")
    if response_start_index != -1:
        return full_response[response_start_index + len("### Assistant:"):].strip()
    else:
        return full_response


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
    offload_folder="offload",
    offload_state_dict=True,
    device_map=device,
    torch_dtype=torch.bfloat16
)


def read_file(file_name):
    with open(file_name, "r") as file:
        text = file.read()
    return text


def generate_text(prompt, llm_model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextStreamer(tokenizer)
    input_ids = inputs.input_ids.to(llm_model.device)
    attention_mask = inputs.attention_mask.to(llm_model.device)
    output = llm_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        max_new_tokens=3000,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
        # new or updated parameter for this model
        num_beams=1,  # set to 1 because it does not work with `streamer`
        no_repeat_ngram_size=4,
        early_stopping=True
    )
    print("\n")
    return extract_response_content(
        tokenizer.decode(output[0], skip_special_tokens=True)
    )

res = ""
while True:
    user_question = input("Please enter your question: ")
    if user_question.lower() == 'exit':
        break
    prompt = get_prompt_with_template(message=user_question)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("start")
    output = generate_text(prompt, model, tokenizer)
    res += output
    print("end")
