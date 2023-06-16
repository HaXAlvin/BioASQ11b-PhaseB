import json
from schema import Schema, And, Use

import os
import bert_score
import openai
from openai.error import RateLimitError, APIError, ServiceUnavailableError
from tqdm import tqdm
from transformers import logging
import time
from multiprocessing import Pool, TimeoutError
from functools import wraps
import traceback

logging.set_verbosity_error()
openai.api_key = "YOUR_API_KEY"
# MODEL = "gpt-3.5-turbo"
MODEL = "gpt-4"
BATCH = 4
SUBMIT = 5
MAX_SNIPPET_LEN = 300
TEMPERATURE = 1
# dataset_path = "/Users/alvin/Projects/BioASQ11-Task b/data/BioASQ-training11b/training11b.json"
dataset_path = "/Users/alvin/Projects/BioASQ11-Task b/data/BioASQ-task11bPhaseB-testset4.txt"

with open(dataset_path, "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)["questions"]

SCHEMA = {
    "yesno": Schema({
        "exact_answer": And(
            str,
            Use(str.lower),  # lowercase the YES/NO
            lambda s: s in ('yes', 'no')
        ),
        "ideal_answer": And(Use(str), lambda s: s.strip() != "", lambda s: len(s.split()) <= 200)
    }),
    "list": Schema({
        "exact_answer": And(
            list,
            lambda x: 100 >= len(x) > 0,  # no more than 100 entries
            lambda x: all((100 >= len(item) > 0) and isinstance(item, str) for item in x),  # no more than 100 characters each
            Use(lambda x: [[i] for i in x])  # convert to list of list
        ),
        "ideal_answer": And(Use(str), lambda s: s.strip() != "", lambda s: len(s.split()) <= 200)
    }),
    "summary": Schema({
        "ideal_answer": And(Use(str), lambda s: s.strip() != "", lambda s: len(s.split()) <= 200)
    }),
    "factoid": Schema({
        "exact_answer": And(
            list,
            # Use(lambda x: x[:5]),  # TODO: select top 5 entries
            lambda x: 5 >= len(x) > 0,  # no more than 5 entries
            Use(lambda x: [[i] for i in x])  # convert to list of list
        ),
        "ideal_answer": And(Use(str), lambda s: s.strip() != "", lambda s: len(s.split()) <= 200)
    })
}

PROMPT = {
    "yesno": """You can only use JSON format to answer my questions. The format must be {"exact_answer":"", "ideal_answer":""}, where exact_answer should be "yes" or "no", and ideal_answer is a short conversational response starting with yes/no then follow on the explain. You should read the chat history's content before answer the question.  The first question is: """,
    "list": """You can only use JSON format to answer my questions. The format must be {"exact_answer":[], "ideal_answer":""}, where exact_answer is a list of precise key entities to answer the question, and ideal_answer is a short conversational response containing an explanation. You should read the chat history's content before answer the question. The first question is: """,
    "summary": """Reply to the answer clearly and easily in less than 3 sentences. You should read the chat history's content before answer the question. The first question is: """,
    "factoid": """You can only use JSON format to answer my questions. The format must be {"exact_answer":[], "ideal_answer":""}. where exact_answer is a list of precise key entities to answer the question. ideal_answer is a short conversational response containing an explanation. You should read the chat history's content before answer the question. The first question is: """
}


def make_message(role, content):
    return {"role": role, "content": content}

def gpt_api_retry(func):
    @wraps(func)
    def warp_func(**kwargs):
        log = ""
        for i in range(5):
            try:
                return func(**kwargs)
            except (RateLimitError, APIError, ServiceUnavailableError, TimeoutError) as e:
                message = f"Retry: {i+1} times, error: {traceback.format_exc()}\n\n"
                tqdm.write(message)
                log += message
                time.sleep(3)
        raise TimeoutError(f"Failed to get response from OpenAI API after 5 retries.\n\n{log}")
    return warp_func


@gpt_api_retry
def completions_with_backoff(**kwargs):
    with Pool(processes=1) as pool:
        process = pool.apply_async(openai.ChatCompletion.create, kwds=kwargs)
        return process.get(timeout=60)


def summary_snippet(snippet):
    messages = [make_message("user", f"Conclusion and summarize this context in less than {MAX_SNIPPET_LEN} letters:\"\"\"{snippet}\"\"\"")]
    completion = completions_with_backoff(model=MODEL, messages=messages, temperature=TEMPERATURE)
    resp = completion.choices[0].message.content
    assert isinstance(resp, str) and resp.strip() != "", f"summary_snippet failed: {resp}"
    return resp


def get_question_answer(q):
    resp = None
    try:  # get result from gpt request
        messages = []
        for sni in q["snippets"]:
            snippet = summary_snippet(sni["text"]) if len(sni["text"]) > MAX_SNIPPET_LEN else sni["text"]
            messages.append(make_message("assistant", snippet))
        # TODO: select top n snippets?
        messages += [make_message("user", PROMPT[q["type"]]), make_message("user", q["body"])]
        completion = completions_with_backoff(model=MODEL, messages=messages, temperature=TEMPERATURE)
        resp = completion.choices[0].message.content.strip(".。\"'")
        result = json.loads(resp) if q["type"] != "summary" else {"ideal_answer": resp}
        result = SCHEMA[q["type"]].validate(result)
        return result
    except Exception as e:
        tqdm.write("====Error, check error.txt====")
        with open("error.txt", "w", encoding="utf-8") as f:
            f.write(f"{str(e)}\nid:{q['id']}\ntype:{q['type']}\nresp:{resp}\n")
        raise e


def request_gpt(specific_q=None):
    loop = tqdm(QUESTIONS) if specific_q is None else [specific_q]
    folder = f"gpt_result/11b_batch_{BATCH}_submit_{SUBMIT}"
    if not os.path.isdir(folder):
        os.mkdir(folder, mode=0o755)
    for question in loop:
        file_path = f"{folder}/{question['id']}.json"
        if os.path.exists(file_path):  # skip if already exists
            continue
        result = get_question_answer(question)
        # add id and save file
        result = {"id": question["id"], **result}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)


def merge_result_and_question():
    submit_file = {"questions": []}
    folder = f"gpt_result/11b_batch_{BATCH}_submit_{SUBMIT}"
    for q in QUESTIONS:
        with open(f"{folder}/{q['id']}.json", "r", encoding="utf-8") as f:
            result = json.load(f)
        merge_q = {**q, **result}
        submit_file["questions"].append(merge_q)
    submit_file_path = f"{folder}/submit.json"
    if os.path.exists(submit_file_path):
        input(f"Overwriting {submit_file_path}. Press enter to continue.....")
    with open(submit_file_path, "w") as f:
        json.dump(submit_file, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    request_gpt()
    merge_result_and_question()
