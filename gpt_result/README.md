# Submit files

|    batch    |  submit  |    model     | System name - description |                   note                    |
|:-----------:|:--------:|:------------:|:-------------------------:|:-----------------------------------------:|
| 11b_batch_1 | submit_1 | gpt3.5 turbo |    submit-1 == IISR-1     |               top-5 snippet               |
|      ^      | submit_2 |     gpt4     |    submit-2 == IISR-2     |               top-5 snippet               |
|      ^      | submit_3 |   biobert    |    submit-3 == IISR-3     |     training on 10b, no exact answer      |
| 11b_batch_2 | submit_1 | gpt3.5 turbo |    submit-1 == IISR-1     |               top-5 snippet               |
|      ^      | submit_2 |     gpt4     |    submit-2 == IISR-2     |               top-5 snippet               |
|      ^      | submit_3 |   biobert    |    submit-3 == IISR-3     | training on 10b, empty fill with submit-2 |
| 11b_batch_3 | submit_1 | gpt3.5 turbo |    submit-1 == IISR-1     |     all snippet, summary in 250 words     |
|      ^      | submit_2 |     gpt4     |    submit-2 == IISR-2     |     all snippet, summary in 250 words     |
|      ^      | submit_3 |   biobert    |    submit-3 == IISR-3     | training on 11b, empty fill with submit-2 |