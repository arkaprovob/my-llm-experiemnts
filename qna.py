import os

import torch
import transformers
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

model_id = "mistralai/Mistral-7B-Instruct-v0.2" # microsoft/phi-2 #mistralai/Mistral-7B-Instruct-v0.2
# #HuggingFaceH4/zephyr-7b-beta

# create a prompt template for model HuggingFaceH4/zephyr-7b-beta


prompt_template = \
    '''
<s>[INST] You are a helpful assistant.
Use the following context to Answer the question below briefly and keep it short and simple:

{context}

{question} [/INST] </s>
    '''

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=False,
    use_auth_token=os.getenv("HF_ACCESS_TOKEN")
)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=False,
    config=model_config,
    quantization_config=bnb_config,
    device_map="cuda:0",
    use_auth_token=os.getenv("HF_ACCESS_TOKEN"),
    torch_dtype=torch.bfloat16
)


def read_file(file_name):
    with open(file_name, "r") as file:
        text = file.read()
    return text


def generate_text(prompt, context, llm_model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(llm_model.device)
    attention_mask = inputs.attention_mask.to(llm_model.device)
    outputs = llm_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_length=2048,
        max_new_tokens=2048,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(context):]


context = read_file("output/summary_tnvft.txt")

while True:
    user_question = input("Please enter your question: ")
    if user_question.lower() == 'exit':
        break
    prompt = prompt_template.format(context=context, question=user_question)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(generate_text(prompt,context, model, tokenizer))



# prompt = prompt_template.format(context=context,
#                                 question="which one should i pick for deep learning "
#                                          "between RTX3090 and A5000?")
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# print(generate_text(prompt,context, model, tokenizer))
