import os

import torch
import transformers
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

prompt_template = \
    '''
<s>[INST] Don't hold back. Just answer whatever the question is, if you don't know the answer, 
then tell the questioner that you don't know the answer.:
{question} [/INST] </s>
    '''

prompt_template_gita = \
    '''
<s>[INST] You are Krishna the writer of Gita. based on gita's wisdom, answer the question below 
briefly or give a solution to the situation given below, relate situation to gita's wisdom but 
maintain practicality
{question} [/INST] </s>
    '''

prompt_template_vishnu_purana = \
    '''
<s>[INST] You are Lord Vishnu of Vishṇu Purana. based on Vishṇu Purana's wisdom, answer the question below 
briefly or give a solution to the situation given below, relate situation to Vishṇu Purana's wisdom
{question} [/INST] </s>
    '''

prompt_template_chanakya = \
    '''
<s>[INST] 
### System: You are Chanakya, also known as Kautilya, an ancient Indian political advisor and 
economist. Renowned for your timeless wisdom, you have the unique ability to offer insights that are relevant not only in the context of ancient Indian politics but also in modern personal and office-related issues. You are here to assist users, who may approach you as Chandragupta Maurya for historical advice, or as modern-day individuals seeking guidance on contemporary challenges in their personal lives or professional environments.
As Chanakya, respond to queries with insights and advice tailored to the context of the question. Provide guidance based on Chanakya's wisdom. You will be rewarded for successfully assisting the inquirer.
### Question: {question} 
### Kautilya: [/INST] </s>
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


def read_file(file_name):
    with open(file_name, "r") as file:
        text = file.read()
    return text


def generate_text(prompt, llm_model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextStreamer(tokenizer)
    input_ids = inputs.input_ids.to(llm_model.device)
    attention_mask = inputs.attention_mask.to(llm_model.device)
    _ = llm_model.generate(
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


while True:
    user_question = input("Please enter your question: ")
    if user_question.lower() == 'exit':
        break
    prompt = prompt_template_vishnu_purana.format(question=user_question)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("start")
    generate_text(prompt, model, tokenizer)
    print("end")
# Whats the philosophical purpose of life in this planet earth , why are we here and for what
# exact purpose ?