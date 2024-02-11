# pip install -q transformers
from transformers import pipeline

checkpoint = "MBZUAI/LaMini-GPT-1.5B"

model = pipeline('text-generation', model = checkpoint,device="cpu")

instruction = 'Please let me know your thoughts on the given place and why you think it deserves to be visited: \n"Barcelona, Spain"'

input_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"


while True:
    user_question = input("Please enter your question: ")
    if user_question.lower() == 'exit':
        break
    generated_text = model(user_question, max_length=512, do_sample=True)[0][
        'generated_text']
    print("Response", generated_text)
