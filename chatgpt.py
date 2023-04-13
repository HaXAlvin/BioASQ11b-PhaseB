import json
from schema import Schema, And, Use

import os
import backoff
import bert_score
import openai
from openai.error import RateLimitError
from tqdm import tqdm
from transformers import logging

logging.set_verbosity_error()
openai.api_key = "YOUR_API_KEY"
MODEL = "gpt-3.5-turbo"
# MODEL = "gpt-4"
BATCH = 0
SUBMIT = 0
MAX_SNIPPET_LEN = 250
dataset_path = "/Users/alvin/Projects/BioASQ11-Task b/data/BioASQ-training11b/training11b.json"
# dataset_path = "/Users/alvin/Projects/BioASQ11-Task b/data/BioASQ-task11bPhaseB-testset1.txt"


SCHEMA = {
    "yesno": Schema({
        "exact_answer": And(Use(str), Use(str.lower), lambda s: s in ('yes', 'no')),
        "ideal_answer": And(Use(str), lambda s: s.strip() != "")
    }),
    "list": Schema({
        "exact_answer": And(list, lambda l: 100 >= len(l) > 0, lambda l: all(100 >= len(i) > 0 for i in l)),
        "ideal_answer": And(Use(str), lambda s: s.strip() != "")
    }),
    "summary": Schema({
        "ideal_answer": And(Use(str), lambda s: s.strip() != ""),
    }),
    "factoid": Schema({
        "exact_answer": And(list, Use(lambda x: x[:5]), lambda l: 5 >= len(l) > 0),
        "ideal_answer": And(Use(str), lambda s: s.strip() != "")
    })
}

PROMPT = {
    "yesno": """You can only use JSON format to answer my questions. The format must be {"exact_answer":"", "ideal_answer":""}, where exact_answer should be "yes" or "no", and ideal_answer is a short conversational response starting with yes/no then follow on the explain. The first question is: """,
    "list": """You can only use JSON format to answer my questions. The format must be {"exact_answer":[], "ideal_answer":""}, where exact_answer is a list of precise key entities to answer the question, and ideal_answer is a short conversational response containing an explanation. The first question is: """,
    "summary": """Reply to the answer clearly and easily in less than 3 sentences. The first question is: """,
    "factoid": """You can only use JSON format to answer my questions. The format must be {"exact_answer":[], "ideal_answer":""}. where exact_answer is a list of precise key entities to answer the question. ideal_answer is a short conversational response containing an explanation.  The first question is: """
}


def make_message(role, content):
    return {"role": role, "content": content}


def average(scores: list):
    return sum(scores) / len(scores)


@backoff.on_exception(backoff.expo, RateLimitError, max_time=60, max_tries=5)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


with open(dataset_path, "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)["questions"]


def request_gpt(specific_q=None):
    loop = tqdm(QUESTIONS) if specific_q is None else [specific_q]
    try:
        for i, q in enumerate(loop):
            # print(q['id'])
            path = f"gpt_result/11b_batch_{BATCH}_submit_{SUBMIT}/{q['id']}.json"
            if os.path.exists(path):
                continue
            messages = [make_message("assistant", sni["text"][:MAX_SNIPPET_LEN]) for sni in q["snippets"]]
            messages = messages[:5] + [make_message("user", PROMPT[q["type"]]), make_message("user", q["body"])]
            # print(messages)
            completion = completions_with_backoff(model=MODEL, messages=messages, temperature=0.7)
            resp = completion.choices[0].message.content.strip(".。\"'")
            resp_text = json.loads(resp) if resp[0] == "{" else {"ideal_answer": resp}
            SCHEMA[q["type"]].validate(resp_text)
            result = {"id": q["id"], **resp_text}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(e)
        print(q["id"], q["type"])
        print(resp_text)
        with open("error.txt", "w", encoding="utf-8") as f:
            f.write(f"{str(e)}\n\nProcess:{i}")


def extract_answer(question, have_answer=False):
    temp_answers = []
    temp_predicts = []

    with open(f"gpt_result/11b_batch_{BATCH}_submit_{SUBMIT}/{question['id']}.json", "r", encoding="utf-8") as f:
        result = json.load(f)
    if have_answer:
        if question["type"] == "yesno":
            temp_answers.append(question["exact_answer"])
            temp_predicts.append(result["exact_answer"])
            temp_answers += question["ideal_answer"]
            temp_predicts += [result["ideal_answer"]] * len(question["ideal_answer"])
        elif question["type"] == "list":
            temp_answers += [" ".join(a) for a in question["exact_answer"]]
            temp_predicts += [" ".join(sorted(result["exact_answer"]))] * len(question["exact_answer"])
            temp_answers += question["ideal_answer"]
            temp_predicts += [result["ideal_answer"]] * len(question["ideal_answer"])
        elif question["type"] == "factoid":
            temp_answers += question["exact_answer"]
            temp_predicts += [result["exact_answer"][0]] * len(question["exact_answer"])
            temp_answers += question["ideal_answer"]
            temp_predicts += [result["ideal_answer"][0]] * len(question["ideal_answer"])
        elif question["type"] == "summary":
            temp_answers += question["ideal_answer"]
            temp_predicts += [result["ideal_answer"]] * len(question["ideal_answer"])
    else:
        if question["type"] == "yesno":
            temp_predicts.append(result["exact_answer"])
            temp_predicts += [result["ideal_answer"]]
        elif question["type"] == "list":
            temp_predicts += sorted(result["exact_answer"])
            temp_predicts += [result["ideal_answer"]]
        elif question["type"] == "factoid":
            temp_predicts += result["exact_answer"]
            temp_predicts += result["ideal_answer"]
        elif question["type"] == "summary":
            temp_predicts.append(result["ideal_answer"])
        temp_answers = [i for i in temp_predicts]
    return temp_answers, temp_predicts


def remove_empty_str(question, list_key):
    if isinstance(question.get(list_key), list):
        question[list_key] = [t for t in question[list_key] if t]
    return question


def clean_and_check_result():
    all_answers = []
    all_predicts = []
    for q in tqdm(QUESTIONS, mininterval=2):
        q = remove_empty_str(q, "exact_answer")
        q = remove_empty_str(q, "ideal_answer")
        for _ in range(10):  # max retry 10 times
            try:
                temp_answers, temp_predicts = extract_answer(q)
                assert len(temp_answers) == len(temp_predicts)
                for i, (a, p) in enumerate(zip(temp_answers, temp_predicts, strict=True)):
                    assert isinstance(a, str) and isinstance(p, str), f"{q['id']},{a},{p}"
                    assert a != "" and p != "", "should not be empty"
                    temp_answers[i] = temp_answers[i][:256]
                    temp_predicts[i] = temp_predicts[i][:256]
                all_answers += temp_answers
                all_predicts += temp_predicts
                break
            except Exception as e:  # retry, request again
                print(e)
                request_gpt(q)
        else:
            raise TimeoutError(f"Too many requests {q['id']}")
    return all_answers, all_predicts


def calc_score():
    all_answers, all_predicts = clean_and_check_result()
    p = []
    r = []
    f = []
    for i in tqdm(range(len(all_answers) // 128)):
        precision, recall, f1 = bert_score.score(all_predicts[i * 128: (i + 1) * 128], all_answers[i * 128: (i + 1) * 128], lang="en", verbose=False, model_type="/Users/alvin/Projects/BioASQ11-Task b/bioasq10b/ideal answer/pretrained_models/biobert_mnli", num_layers=5, batch_size=128)
        p.append(precision.mean().item())
        r.append(recall.mean().item())
        f.append(f1.mean().item())
    print(average(p), average(r), average(f))
    # 0.6891707161377216 0.7301315315838518 0.700919523321349


def merge_result_and_question():
    submit_file = {"questions": []}
    for q in QUESTIONS:
        with open(f"gpt_result/11b_batch_{BATCH}_submit_{SUBMIT}/{q['id']}.json", "r", encoding="utf-8") as f:
            result = json.load(f)
        # if q["type"] == "list":
        #     if len(result["exact_answer"]) == 1:
        #         result["exact_answer"][0] = result["exact_answer"][0].strip(".")
        #         if "and" in result["exact_answer"][0]:
        #             result["exact_answer"] = result["exact_answer"][0].split("and")
        #         if "," in result["exact_answer"][0]:
        #             result["exact_answer"] = result["exact_answer"][0].split(",")
        #     temp = []
        #     for i in result["exact_answer"]:
        #         if not i.strip():
        #             continue
        #         temp.append([i.strip()])
        #     result["exact_answer"] = temp
        # if q["type"] == "factoid":
        #     result["ideal_answer"] = result["ideal_answer"][0]
        # if q["type"] == "yesno":
        #     result["exact_answer"] = result["exact_answer"].lower()
        merge_q = {**q, **result}
        submit_file["questions"].append(merge_q)
    submit_file_path = f"gpt_result/11b_batch_{BATCH}_submit_{SUBMIT}/submit.json"
    if os.path.exists(submit_file_path):
        input(f"overwriting {submit_file_path}. Press enter to continue.")
    with open(submit_file_path, "w") as f:
        json.dump(submit_file, f, ensure_ascii=False, indent=4)


# calc_score()
# merge_result_and_question()
request_gpt()
# 放在同個dict
# exact_answer ideal_answer分開
# opanai api embedding 算cos相似度
