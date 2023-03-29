import openai
from openai.error import RateLimitError
import json
from tqdm import tqdm
import backoff
import bert_score
from transformers import logging
logging.set_verbosity_error()
openai.api_key = "YOUR_API_KEY"
model = "gpt-3.5-turbo"


prompt = {
    "yesno": """You can only use JSON format to answer my questions. The format must be {"exact_answer":"", "ideal_answer":""}, where exact_answer is either "yes" or "no", and ideal_answer is a short conversational response starting with yes/no then follow on the explain. The first question is: """,
    "list": """You can only use JSON format to answer my questions. The format must be {"exact_answer":[], "ideal_answer":""}, where exact_answer is a non-empty array of precise answers, and ideal_answer is a short conversational response containing an explanation. The first question is: """,
    "summary": """Reply to the answer clearly and easily in less than 3 sentences. The first question is: """,
    "factoid": """You can only use JSON format to answer my questions. The format must be {"exact_answer":[], "ideal_answer":[]}. where ideal_answer is a non-empty list of short texts including prominent supportive information. exact_answer is a list of key entities from ideal_answer to the question.  The first question is: """
}


def make_message(role, content):
    return {"role": role, "content": content}


@backoff.on_exception(backoff.expo, RateLimitError, max_time=60, max_tries=5)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


with open("./data/BioASQ-training11b/training11b.json", "r", encoding="utf-8") as f:
    questions = json.load(f)["questions"]


def request_gpt(specific_q=None):
    if specific_q is not None:
        loop = [specific_q]
    else:
        loop = tqdm(questions)
    try:
        for i, q in enumerate(loop):
            # if i < 4486:
            #     continue
            messages = [make_message("assistant", sni["text"][:250]) for sni in q["snippets"]]
            messages = messages[:5] + [make_message("user", prompt[q["type"]]), make_message("user", q["body"])]
            # print(messages)
            completion = completions_with_backoff(model=model, messages=messages, temperature=0.7)
            resp = completion.choices[0].message.content.strip(".。\"'")
            # print(resp)
            resp_text = json.loads(resp) if resp[0] == "{" else {"ideal_answer": resp}
            resp_text = {k: str(v) for k, v in resp_text.items()}
            result = {"id": q["id"], **resp_text}
            with open(f"gpt_result/{q['id']}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
    except Exception as e:
        with open("error.txt", "w", encoding="utf-8") as f:
            f.write(f"{str(e)}\n\nProcess:{i}")


def extract_answer(question):
    temp_answers = []
    temp_predicts = []
    with open(f"gpt_result/{question['id']}.json", "r", encoding="utf-8") as f:
        result = json.load(f)
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
    return temp_answers, temp_predicts


def calc_score():
    all_answers = []
    all_predicts = []
    p_bar = tqdm(questions, mininterval=2)
    for _, q in enumerate(p_bar):
        while isinstance(q.get("exact_answer", None), list) and "" in q["exact_answer"]:
            q["exact_answer"].remove("")
        while isinstance(q.get("ideal_answer", None), list) and "" in q["ideal_answer"]:
            q["ideal_answer"].remove("")
        for i in range(10):
            try:
                temp_answers, temp_predicts = extract_answer(q)
                assert len(temp_answers) == len(temp_predicts)
                for _, (a, p) in enumerate(zip(temp_answers, temp_predicts)):
                    assert (isinstance(a, str) and isinstance(p, str)), f"{q['id']},{a},{p}"
                    assert a != "" and p != "", "should not be empty"
                all_answers += temp_answers
                all_predicts += temp_predicts
                break
            except Exception as e:
                print(e)
                request_gpt(q)
        else:
            raise TimeoutError(f"Too many requests {q['id']}")
    assert len(all_answers) == len(all_predicts)
    p = []
    r = []
    f = []

    # for ans, pred in tzip(all_answers, all_predicts):
    for i in tqdm(range(len(all_answers)//512)):
        # assert isinstance(pred, str) and isinstance(ans, str), "Not string"
        pred = [pred[:256] for pred in all_predicts[i*512:(i+1)*512]]
        ans = [ans[:256] for ans in all_answers[i*512:(i+1)*512]]
        precision, recall, f1 = bert_score.score(pred, ans, lang="en", verbose=False, model_type="/Users/alvin/Projects/BioASQ11-Task b/bioasq10b/ideal answer/pretrained_models/biobert_mnli", num_layers=5, batch_size=128)
        p.append(precision.mean().item())
        r.append(recall.mean().item())
        f.append(f1.mean().item())
    average = lambda x: sum(x)/len(x)
    print(average(p), average(r), average(f))
    # 0.6891707161377216 0.7301315315838518 0.700919523321349


calc_score()
