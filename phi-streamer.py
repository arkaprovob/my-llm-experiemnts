import os
from threading import Thread

import torch
import transformers
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, StoppingCriteria
import time

model_id = "microsoft/phi-2"

PHI_PROMPT = \
'''
{query}
Answer:
'''


device = "cuda:0"
trust_remote_code = True
cache_dir = "/media/arko/work/.hfcache"


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
    # quantization_config=bnb_config,
    device_map=device,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir
)


# custom stopping criteria (avoid generating hallucinated prompts)
# still includes these tokens in the output but stops generating after them
class Phi2StoppingCriteria(StoppingCriteria):
    def __init__(self):
        stop_list = ["Exercise", "Exercises", "exercises:", "<|endoftext|>", "<eot>"]
        tokenphrases = []
        for token in stop_list:
            tokenz = tokenizer(token, return_tensors="pt").input_ids[0].tolist()
            print(f"word: {token} corresponds to token: {tokenz}")
            tokenphrases.append(
                tokenz
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
        #stopping_criteria=[Phi2StoppingCriteria()],
        streamer=streamer
    )
    end_time = time.time()
    # Check the type and structure of outputs
    if not isinstance(outputs, torch.Tensor) or len(outputs.shape) != 2:
        print("Unexpected output format from the model.")
        return None

    duration = end_time - start_time
    num_tokens = outputs.shape[1]  # Assuming outputs is a 2D tensor with shape [num_sequences, num_tokens]
    tokens_per_second = num_tokens / duration
    return tokens_per_second


while True:
    user_question = input("Please enter your question: ")
    if user_question.lower() == 'exit':
        break
    prompt = PHI_PROMPT.format(query=user_question)
    #prompt = PHI_PROMPT.format(query="what is your name")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False,cache_dir=cache_dir)
    tps = generate_text(prompt, model, tokenizer)
    print(f"Token generation speed: {tps} tokens/second")

