import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import BioGptForCausalLM, BioGptTokenizer

# def seed_everything(seed=1234):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True


class BioGpt_BioASQ(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

    def forward(self, batch):
        return self.model(
            **encoded_input, output_attentions=True, output_hidden_states=True
        )

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


batch = ["What's your name?", "Who are you?"]

model = BioGpt_BioASQ()
output = model(batch)
print(output.keys())
print(output.logits.shape)
