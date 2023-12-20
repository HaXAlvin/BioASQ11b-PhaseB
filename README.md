# NCU-IISR: Prompt Engineering on GPT-4 to Stove Biological Problems in BioASQ 11b Phase B

This is the repository for [this](https://ceur-ws.org/Vol-3497/paper-009.pdf) paper. Including how we prompt and extract the results from ChatGPT via the OpenAI API.

The results are recorded [here](./gpt_result/README.md) or you can go to official [website](http://participants-area.bioasq.org) to check the result.

## To-Do
- [ ] Use new openai api to force return json format
- [ ] Compare between select top-n snippets and summary the snippets
- [ ] Postprocessing when factoid type question have more than 5 entries

## Citation
```bibtex
@inproceedings{hsueh2023bioasq,
  title        = {NCU-IISR: Prompt Engineering on GPT-4 to Stove Biological Problems in BioASQ 11b Phase B},
  author       = {Chun-Yu Hsueh and Yu Zhang and Yu-Wei Lu and Jen-Chieh Han and Wilailack Meesawad and Richard Tzong-Han Tsai},
  year         = 2023,
  booktitle    = {Working Notes of the Conference and Labs of the Evaluation Forum (CLEF 2023), Thessaloniki, Greece, September 18th to 21st, 2023},
  publisher    = {CEUR-WS.org},
  series       = {CEUR Workshop Proceedings},
  volume       = 3497,
  pages        = {114--121},
  url          = {https://ceur-ws.org/Vol-3497/paper-009.pdf},
  editor       = {Mohammad Aliannejadi and Guglielmo Faggioli and Nicola Ferro and Michalis Vlachos}
}
```

<details>
<summary><h2>Working Note</h2></summary>

### Consider Systems
  1. BioBERT[^1]
  2. [BioGPT](https://github.com/microsoft/BioGPT) (GPT2 based)[^2]
  3. ChatGPT (GPT4)
  4. ChatGPT (GPT3.5)
  5. Fine-tune/Transfer GPT3，prompt turning
### Scoring
  - [OpenAI Evals](https://github.com/openai/evals)
  - [BertScore](https://github.com/Tiiiger/bert_score)
  - ROUGE-SU4
### Others
  - training 11b = training 10b + testing 10b
  - [BioLinkBERT](https://github.com/michiyasunaga/LinkBERT)
  - [OPT-175b](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)
  - Retriever method for snippets
    - BM25 & TF-IDF
    - [OpenAI Embedding](https://platform.openai.com/docs/guides/embeddings) + Cos similarity


[^1]: Previous Method [Paper](https://ceur-ws.org/Vol-3180/paper-27.pdf) and [Note](https://hackmd.io/@denis049/rygMFy0Co)
[^2]: Inaccurate result on [PapersWithCode](https://paperswithcode.com/sota/question-answering-on-pubmedqa), should follow [PudMedQA homepage](https://pubmedqa.github.io)
</details>