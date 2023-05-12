# In batch 1 ~ 3, submit system 3 missing exact answer.

import json
import os
batch = 3
submit = 3
source_submit = 2

with open(f"./gpt_result/11b_batch_{batch}_submit_{submit}/BioASQ-task11bPhaseB-testset{batch}_save-model_4.json", "r", encoding="utf-8") as f:
    to_be_replace = json.load(f)

with open(f"./gpt_result/11b_batch_{batch}_submit_{source_submit}/submit.json", "r", encoding="utf-8") as f:
    d = json.load(f)
    source_exact_answer = {i["id"]: i for i in d["questions"]}


for i in to_be_replace["questions"]:
    if i["type"] == "summary":
        del i["exact_answer"]
    elif i["type"] != "yesno":
        i["exact_answer"] = source_exact_answer[i["id"]]["exact_answer"]

path = f"./gpt_result/11b_batch_{batch}_submit_{submit}/submit.json"
if os.path.exists(path):
    print(path)
    input("File exists, press any key to continue")

with open(path, "w", encoding="utf-8") as f:
    json.dump(to_be_replace, f, indent=4)
