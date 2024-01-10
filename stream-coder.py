import os

import torch
import transformers
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer,StoppingCriteria
from prompt import SUPPORT_ENGINEER

model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"


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


class DsaiStoppingCriteria(StoppingCriteria):
    def __init__(self):
        stop_list = ["<end>", "<｜end▁of▁sentence｜>","end▁of▁sentence","end_of_sentence",
                     "<｜end_of_sentence｜>","<|end▁of▁sentence|>","<|end_of_sentence|>"
                     "<｜end▁of▁sentence｜><｜begin▁of▁sentence｜># 100 Days Of Code - Log",
                     "<｜end▁of▁sentence｜><｜begin▁of▁sentence｜>", "requirements", "<|EOT|>"
            ,"<|endoftext|>"]
        tokenphrases = []
        for token in stop_list:
            tokenphrases.append(
                tokenizer(token, return_tensors="pt").input_ids[0].tolist()
            )
        print(tokenphrases)
        self.tokenphrases = tokenphrases

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for tokenphrase in self.tokenphrases:
            if tokenphrase == input_ids[0].tolist()[-len(tokenphrase):]:
                return True

def generate_text(prompt, llm_model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextStreamer(tokenizer)
    input_ids = inputs.input_ids.to(llm_model.device)
    attention_mask = inputs.attention_mask.to(llm_model.device)
    _ = llm_model.generate(
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
        stopping_criteria=[DsaiStoppingCriteria()],
        streamer=streamer
    )


while True:
    user_question = input("Please enter your question: ")
    if user_question.lower() == 'exit':
        break
    prompt = SUPPORT_ENGINEER.format(question=user_question)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("start")
    generate_text(prompt, model, tokenizer)
    print("end")