import os

import torch
import transformers
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, StoppingCriteria, \
    StoppingCriteriaList, GenerationConfig
from prompt import SUPPORT_ENGINEER, SUPPORT_ENGINEER2, SUPPORT_ENGINEER3

cache_dir = "/media/arko/work/.hfcache"
# model_id = "upstage/SOLAR-10.7B-Instruct-v1.0"
# model_id = "Nexusflow/NexusRaven-V2-13B"
model_id = "deepseek-ai/deepseek-moe-16b-chat"

device = "auto"
trust_remote_code = True

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
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
    device_map=device,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir
)
model.generation_config = GenerationConfig.from_pretrained(model_id, cache_dir=cache_dir)
model.generation_config.pad_token_id = model.generation_config.eos_token_id


def read_file(file_name):
    with open(file_name, "r") as file:
        text = file.read()
    return text


class DsaiStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        stop_list = ["<end>", "else"]
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


tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
stopping_criteria = StoppingCriteriaList([DsaiStoppingCriteria(tokenizer)])


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
        # stopping_criteria=stopping_criteria,
        streamer=streamer
    )


while True:
    user_question = input("Please enter your question: ")
    if user_question.lower() == 'exit':
        break
    prompt = SUPPORT_ENGINEER.format(question=user_question)
    print("start")
    generate_text(prompt, model, tokenizer)
    print("end")
