import json

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import BioGptTokenizer


class BioASQRawData:
    def __init__(self, split_size=None):
        if split_size is None:
            split_size = [0.8, 0.1, 0.1]
        with open("./data/BioASQ-training11b/training11b.json", "r") as f:
            questions = json.load(f)["questions"]
        print(questions[0])
        self.split_size = split_size
        self.questions = list(
            map(
                lambda q: {
                    "body": q["body"],  # str
                    "ideal_answer": q["ideal_answer"],  # [str]
                    "exact_answer": [q["exact_answer"]]
                    if q["type"] == "yesno"
                    else q.get(
                        "exact_answer", None
                    ),  # [str] or None, summary will not have exact ans
                    "snippets": list(
                        map(lambda snippet: snippet["text"], q["snippets"])
                    ),  # [str]
                    "type": q["type"],  # str
                },
                questions,
            )
        )

    def get_data(self):
        return self.questions


BioASQRawData()


class BioASQDataset(Dataset):
    def __init__(self, split_set: str = "train"):
        assert split_set in ["train", "dev", "test"]

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 1
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.train_dataset = BioASQDataset(split_set="train")
        self.dev_dataset = BioASQDataset(split_set="dev")
        self.test_dataset = BioASQDataset(split_set="test", is_test=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, collate_fn=self.collate_fn
        )

    def collate_fn(self, batch_data):
        encoded_input = self.tokenizer(
            batch_data, return_tensors="pt", padding=True, truncation=True
        )
        return encoded_input


# print(data.keys())
