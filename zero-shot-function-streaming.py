import os
from threading import Thread

import torch
import transformers
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import time

model_id = "Nexusflow/NexusRaven-V2-13B"

RAVEN_PROMPT = \
'''
Function:
def fetch_user_id_by_slack_name(slack_name):
    """
    Fetches the user id by slack name.

    :param slack_name: The slack name of the user.
    :return: The user id.
    """
    
Function:
def detect_app_and_environment_from_url(url):
    """
    Extracts the app name and environment from the url. which can be used as an input to function 
    get_app_details to get the app details.

    :param url: The url to extract the app details from.
    :return: A tuple containing the app name and environment.
    """
    
Function:
def get_app_details(app_env_tuple):
    """
    Extracts the app details from the app name and environment.

    :param app_env_tuple: A tuple containing the app name and environment.
    :return: A dictionary containing the app details.

    The returned dictionary contains the following keys:
    - "environment": The environment name of the app.
    - "app_name": The name of the app.
    - "access": The access permissions for the app. It's a list containing 'read', 'write', 'delete'.
    - "health_check": The URL for the health check endpoint of the app.
    - "log_file_location": The location of the log file for the app.
    - "app_path": The path of the app.
    - "app_type": The type of the app. It can be "java", "python" "php", "go" etc.
    - "approvals": The approval requirements for different actions. It's a list of dictionaries.
    - "app_owner": The owner of the app.
    - "repo_url": The source code url of the repository where the app's code is stored.
    - "issues": The issue tracking system used for the app.
    - "docs": The location of the documentation for the app.
    """
    
Function:
def read_last_n_log_lines(app_details, n=50):
    """
    Extracts the last n lines from the log file of a given application.

    :param app_details: A dictionary containing the application details.
    :param n: The number of lines to extract from the end of the log file.
    :return: The last n lines from the log file as a string.
    """
    
Function:
def debug_problem(log_file_contents, source_code_url):
    """
    Analyzes the log details and source code to debug the problem.

    :param log_file_contents: The log file contents.
    :param source_code_url: The source code url.
    :return: The analysis results.
    """

Function:
def list_jira_issues(issue_space):
    """
    Lists the jira issues.

    :param issue_space: The type of the issue.
    :return: The list of issues.
    """

Function:
def create_jira_issue(issue_space, issue_type, issue_summary, issue_description):
    """
    Creates a jira issue.

    :param issue_space: The type of the issue.
    :param issue_type: The type of the issue.
    :param issue_summary: The summary of the issue.
    :param issue_description: The description of the issue.
    :return: The created issue.
    """
    
Function:
def check_app_health(application_details):
    """
    Checks the app health.

    :param application_details: the application details which is a dictionary and contains the
    following keys:
    - "environment": The environment name of the app.
    - "app_name": The name of the app.
    - "access": The access permissions for the app. It's a list containing 'read', 'write', 'delete'.
    - "health_check": The URL for the health check endpoint of the app.
    - "log_file_location": The location of the log file for the app.
    - "app_path": The path of the app.
    - "app_type": The type of the app. In this case, it's "java".
    - "approvals": The approval requirements for different actions. It's a list of dictionaries.
    - "app_owner": The owner of the app.
    - "repo_url": The source code url of the repository where the app's code is stored.
    - "issues": The issue tracking system used for the app.
    - "docs": The location of the documentation for the app.
    
    :return: The health check results.
    """
    
Function:
def restart_app(application_details):
    """
    Restarts the app by running restart script. then performs health check. If health check
    status is not 200, then it will create a jira issue. with issue_space=codeship, issue_type=bug,
    issue_summary="unable to restart the application",
    issue_description=issue_summary+read_last_n_log_lines(app_name)

    :param application_details: the application details which is a dictionary and contains the
    following keys:
    - "environment": The environment name of the app.
    - "app_name": The name of the app.
    - "access": The access permissions for the app. It's a list containing 'read', 'write', 'delete'.
    - "health_check": The URL for the health check endpoint of the app.
    - "log_file_location": The location of the log file for the app.
    - "app_path": The path of the app.
    - "app_type": The type of the app. In this case, it's "java".
    - "approvals": The approval requirements for different actions. It's a list of dictionaries.
    - "app_owner": The owner of the app.
    - "repo_url": The source code url of the repository where the app's code is stored.
    - "issues": The issue tracking system used for the app.
    - "docs": The location of the documentation for the app.

    :return: The restart results.
    """

User Query: {query}<human_end>

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
    streamer = TextStreamer(tokenizer)
    input_ids = inputs.input_ids.to(llm_model.device)
    attention_mask = inputs.attention_mask.to(llm_model.device)

    start_time = time.time()
    outputs = llm_model.generate(
        input_ids=input_ids,
        #attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=2048,
        #top_k=50,
        #top_p=0.95,
        temperature=0.001,
        #num_return_sequences=1,
        #no_repeat_ngram_size=3,
        #pad_token_id=tokenizer.eos_token_id,
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
    prompt = RAVEN_PROMPT.format(query=user_question)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tps = generate_text(prompt, model, tokenizer)
    print(f"Token generation speed: {tps} tokens/second")

