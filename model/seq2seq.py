import random
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(pl.LightningModule):
    def __init__(
        self, config, train_iter, val_iter, test_iter, src_vocab_size, trg_vocab_size
    ):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        input_dim = src_vocab_size
        output_dim = trg_vocab_size
        embedding_dim = self.config.model.embedding_dim
        enc_hidden_dim = self.config.model.enc_hidden_dim
        dec_hidden_dim = self.config.model.dec_hidden_dim
        n_layers = self.config.model.n_layers
        dropout_ratio = self.config.model.dropout_ratio

        self.encoder = Encoder(
            input_dim, embedding_dim, enc_hidden_dim, n_layers, dropout_ratio
        )
        self.decoder = Decoder(
            output_dim, embedding_dim, dec_hidden_dim, n_layers, dropout_ratio
        )

    def forward(self, src, trg, teacher_force_ratio=0.5):
        hidden, cell = self.encoder(src)

        decoder_input = trg[0, :]
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        for i in range(1, trg_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            outputs[i] = output
            top1 = output.argmax(1)

            teacher_force = random.random() < teacher_force_ratio
            decoder_input = trg[i] if teacher_force else top1

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(batch.src, batch.trg, 0.5)
        output_dim = outputs.shape[-1]

        outputs = outputs[1:].view(-1, output_dim)
        trg = batch.trg[1:].view(-1)
        loss = F.cross_entropy(outputs, trg)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        self.log("train_loss", loss, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch.src, batch.trg, 0.5)
        output_dim = outputs.shape[-1]
        outputs = outputs[1:].view(-1, output_dim)
        trg = batch.trg[1:].view(-1)
        val_loss = F.cross_entropy(outputs, trg)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.7)
        # return torch.optim.Adam(self.parameters())

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        # warm up lr
        warm_up_epoch = 4
        if epoch > warm_up_epoch:

            if (
                self.trainer.global_step % self.trainer.num_training_batches == 0
                or self.trainer.global_step % self.trainer.num_training_batches
                == self.trainer.num_training_batches // 2
            ):
                lr_scale = 0.5

                for pg in optimizer.param_groups:
                    pg["lr"] = (
                        lr_scale * self.trainer.optimizers[0].param_groups[0]["lr"]
                    )

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.val_iter

    # def test_dataloader(self):
    # return self.test_iter


class Encoder(nn.Module):
    def __init__(
        self, input_dim, embedding_dim, enc_hidden_dim, n_layers, dropout_ratio
    ):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim, enc_hidden_dim, n_layers, dropout=dropout_ratio
        )
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, output_dim, embedding_dim, dec_hidden_dim, n_layers, dropout_ratio
    ):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim, dec_hidden_dim, n_layers, dropout=dropout_ratio
        )
        self.fc = nn.Linear(dec_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, trg, hidden, cell):
        trg = trg.unsqueeze(0)
        embedded = self.dropout(self.embedding(trg))
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc(outputs.squeeze(0))
        return prediction, hidden, cell
