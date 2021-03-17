import random
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqAttention(pl.LightningModule):
    def __init__(
        self, config, train_iter, val_iter, test_iter, src_vocab_size, trg_vocab_size
    ):
        super(Seq2SeqAttention, self).__init__()
        self.config = config
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        input_dim = src_vocab_size
        output_dim = trg_vocab_size
        embedding_dim = self.config.model.embedding_dim
        enc_hidden_dim = self.config.model.hidden_dim
        dec_hidden_dim = self.config.model.hidden_dim
        dropout_ratio = self.config.model.dropout_ratio

        self.attn = Attention(enc_hidden_dim, dec_hidden_dim)
        self.encoder = Encoder(
            input_dim, embedding_dim, enc_hidden_dim, dec_hidden_dim, dropout_ratio
        )
        self.decoder = Decoder(
            output_dim,
            embedding_dim,
            enc_hidden_dim,
            dec_hidden_dim,
            dropout_ratio,
            self.attn,
        )

    def forward(self, src, trg, teacher_force_ratio=0.5):
        enc_outputs, hidden = self.encoder(src)

        decoder_input = trg[0, :]

        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        for i in range(1, trg_len):
            output, hidden = self.decoder(decoder_input, hidden, enc_outputs)

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
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.val_iter

    # def test_dataloader(self):
    # return self.test_iter


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        enc_hidden_dim,
        dec_hidden_dim,
        dropout_ratio,
    ):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, enc_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        output, hidden = self.rnn(embedded)
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )

        return output, hidden


class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, enc_outputs):
        src_len = enc_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, enc_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embedding_dim,
        enc_hidden_dim,
        dec_hidden_dim,
        dropout_ratio,
        attention,
    ):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.fc_out = nn.Linear(
            (enc_hidden_dim * 2) + dec_hidden_dim + embedding_dim, output_dim
        )
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input, hidden, enc_outputs):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))
        attention = self.attention(hidden, enc_outputs)
        attention = attention.unsqueeze(1)

        enc_outputs = enc_outputs.permute(1, 0, 2)

        weighted = torch.bmm(attention, enc_outputs)
        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden.squeeze(0)
