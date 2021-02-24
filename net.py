import random
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(pl.LightningModule):
    def __init__(self, train_iter, val_iter, test_iter, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        encoder = Encoder(src_vocab_size)
        decoder = Decoder(tgt_vocab_size)
        self.seq2seq = Seq2Seq(encoder, decoder)

    def forward(self, src, tgt):
        outputs = self.seq2seq(src, tgt)

        return outputs

    def training_step(self, batch, batch_idx):
        output = self(batch.src, batch.tgt)

        output = output[1:].view(-1, output.shape[-1])
        tgt = batch.tgt[1:].view(-1)

        # print("### output: ", output)
        # print("### tgt:", tgt)

        loss = F.cross_entropy(output, tgt)

        self.log("train_loss", loss, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch.src, batch.tgt)

        print("##output:", output.size())
        print("##tgt:", batch.tgt.size())
        output = output[1:].view(-1, output.shape[-1])
        tgt = batch.tgt[1:].view(-1)

        loss = F.cross_entropy(output, tgt)

        self.log("val_loss", loss, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.7)

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.val_iter

    # def test_dataloader(self):
    #     return self.test_iter


class Encoder(nn.Module):
    def __init__(self, src_vocab_size):
        super().__init__()

        input_dim = src_vocab_size
        emb_dim = 1000
        hid_dim = 1000
        n_layers = 4

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers)

    def forward(self, src):

        x = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(x)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size):
        super().__init__()
        self.output_dim = tgt_vocab_size
        hid_dim = 1000
        n_layers = 4
        emb_dim = 1000

        self.embedding = nn.Embedding(self.output_dim, hid_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers)
        self.fc = nn.Linear(hid_dim, self.output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)

        x = self.embedding(input)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output.squeeze(0))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):

        teacher_forcing_ratio = 0.5
        tgt_len = tgt.shape[0]
        batch_size = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size)
        hidden, cell = self.encoder(src)
        input = tgt[0, :]

        for i in range(1, tgt_len):

            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[i] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input = tgt[i] if teacher_force else top1

        return outputs