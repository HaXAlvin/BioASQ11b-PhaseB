# Note

## Consider Systems
  1. BioBERT[^1]
  2. [BioGPT](https://github.com/microsoft/BioGPT) (GPT2 based)[^2]
  3. ChatGPT (GPT4)
  4. ChatGPT (GPT3.5)
  5. Fine-tune/Transfer GPT3，prompt turning
## Scoring
  - [OpenAI Evals](https://github.com/openai/evals)
  - [BertScore](https://github.com/Tiiiger/bert_score)
  - ROUGE-SU4
## Others
  - training 11b = training 10b + testing 10b
  - [BioLinkBERT](https://github.com/michiyasunaga/LinkBERT)
  - [OPT-175b](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)
  - 取 snippets 的方法
    - BM25
    - TF-IDF
    - [OpenAI Embedding](https://platform.openai.com/docs/guides/embeddings) + Cos相似度
  - [Submit History](./gpt_result/README.md)


[^1]: 去年的模型
      - [Paper](https://ceur-ws.org/Vol-3180/paper-27.pdf)
      - [Denis Note](https://hackmd.io/@denis049/rygMFy0Co)
[^2]: 學長說在 PudMedQA 上的成績有問題，可能需要查證



<!-- vscode setting -->
<!-- Markdown Preview Enhanced -->
<!-- "markdown-preview-enhanced.enableExtendedTableSyntax": true -->
<!-- "markdown-preview-enhanced.enableCriticMarkupSyntax": true  -->