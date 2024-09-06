import os

import torch
import transformers
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, StoppingCriteria, \
    StoppingCriteriaList, GenerationConfig
from prompt import SUPPORT_ENGINEER, SUPPORT_ENGINEER2, SUPPORT_ENGINEER3

cache_dir = "/media/arko/work/.hfcache"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

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
        max_new_tokens=256,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        # stopping_criteria=stopping_criteria,
        streamer=streamer
    )


PROMPT_TEMPLATE = '''
### Instruction
Read the following text in the context and answer the question. If the answer does not exists in 
the context then don't try to generate the answer, instead write Answer does not exists in the context

### Context
spaship is a platform for deploying, integrating, and managing single-page apps (spas). spaship is an open source project. it is optimized to deploy and host spas in a very efficient and performant way.
confused ... 😕 let me break it down for you 😃
spaship as a product abstracts the underlying deployment infrastructure through a management portal (spaship manager or cli) and cloud-native deployment engine so that the developers can focus on developing spas rather than dealing with deployment/delivery and ci configurations. spaship focuses on improving the deployment experience for developers by providing an easy to use cli tool, an interactive user interface for web property management and with the ability to be integrated with existing ci based deployment platforms.
in short, spaship is designed to make it simple for developers to deploy their spas on any platform (including kubernetes based ones) without having to know the details of the infrastructure setup. the application setup and website registration are just a one-time configuration, do it once and then forget. focus on what you care about the most! the end goal of spaship is to rid developers of everyday pain points such as:
    dealing with infrastructure
    dealing with ci
    long deployment time
    fixed number of environments to work with
what is spa 📌 ?
a single-page application (spa) is an application that is run client-side in a user’s browser. it is loaded with a single page request and the content of the application is updated via javascript. it is generally comprised of static assets such as javascript, css, and html.
spas are all about serving an outstanding ux by trying to imitate a “natural” environment in the browser — no page reloads, no extra wait time. it is just one web page that you visit which then loads all other content using javascript — which they heavily depend on.
info
so the mission of spaship is to provide delightful single-page app deployment and hosting.
problems spaship tackle ♻️#
    deploying web apps are too difficult. it consumes a lot of developer time.
        manually uploading zip files through drupal.
        creating new openshift origins per spa.
        using static hosting like netstorage.
    development lifecycle is manual, complex, and inefficient.
        running in a real environment (eg, qa, stage, prod)
        ci/cd
    claiming url namespaces at any path requires manual mappings in f5 and akamai
solutions spaship provides 💻 💯#
    efficient deployment of spas.
        deploy apps across any environment in a matter of seconds.
    provide bootstrap, configuration & spa developer experience.
    chroming
        supports apache server side includes(ssi).
    routing & namespace management
    ci/cd, lifecycle & validation
    easily layered on top of existing sites
what spaship isn't .... ❌
    spaship is not limited to front-end tech choices. we can use any tech stack for spas development.
    it is not replacement of create-react-app, vue-cli . it doesn't create a react or vue project setup for us.

### Question
{question}'''

while True:
    user_question = input("Please enter your question: ")
    if user_question.lower() == 'exit':
        break
    prompt = PROMPT_TEMPLATE.format(question=user_question)
    print("start")
    generate_text(prompt, model, tokenizer)
    print("end")
