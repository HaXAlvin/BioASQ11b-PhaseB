import json

with open("./data/BioASQ-training11b/training11b.json", "r") as f:
    questions = json.load(f)["questions"]

count = 0
for q in questions:
    if q["type"] == "summary":
        count += 1
    if count == 12:
        format_question = {
            "body": q["body"],
            "snippets": [snippet["text"] for snippet in q["snippets"]],
        }
        print(format_question)
        print("ideal_answer: ", q["ideal_answer"])
        break
