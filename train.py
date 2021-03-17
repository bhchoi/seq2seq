import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning import loggers
from torchtext.data import Field, TabularDataset, BucketIterator, ReversibleField
from torchtext.datasets import Multi30k
from omegaconf import OmegaConf
import spacy

from model import Seq2Seq, Seq2SeqAttention


def tokenize_de():
    spacy_de = spacy.load("de")

    def tokenize(text):
        return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

    return tokenize


def tokenize_en():
    spacy_en = spacy.load("en")

    def tokenize(text):
        return [token.text for token in spacy_en.tokenizer(text)]

    return tokenize


def main(config):
    # TODO: 데이터셋 aihub로 바꾸기

    # src = Field(init_token="<sos>", eos_token="<eos>", lower=True)
    # tgt = Field(init_token="<sos>", eos_token="<eos>", lower=True)

    # dataset = TabularDataset(
    #     path="data/dataset.tsv", format="tsv", fields=[("src", src), ("tgt", tgt)]
    # )
    # train, val, test = dataset.split(split_ratio=[0.8, 0.1, 0.1])

    # src.build_vocab(train)
    # tgt.build_vocab(train)

    # src_vocab_size = len(src.vocab)
    # tgt_vocab_size = len(tgt.vocab)

    # train_iter, val_iter, test_iter = BucketIterator.splits(
    #     (train, val, test),
    #     batch_size=32,
    #     sort=False,
    # )

    SRC = Field(
        tokenize=tokenize_de(),
        init_token="<sos>",
        eos_token="<eos>",
        lower=True,
    )

    TRG = Field(
        tokenize=tokenize_en(),
        init_token="<sos>",
        eos_token="<eos>",
        lower=True,
    )

    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".de", ".en"), fields=(SRC, TRG)
    )

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    train_iter, valid_iter, test_iter = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=128
    )

    net = Seq2Seq(
        config, train_iter, valid_iter, test_iter, len(SRC.vocab), len(TRG.vocab)
    )

    tb_logger = loggers.TensorBoardLogger(f"{config.log_path}{config.trainer.task}")

    lr_callback = LearningRateMonitor("step")

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoint/{config.trainer.task}",
        filename="{epoch}-{val_loss:.5f}",
    )
    early_stopping_callback = EarlyStopping("val_loss", patience=3)
    trainer = pl.Trainer(
        gpus=1,
        callbacks=[lr_callback, model_checkpoint_callback, early_stopping_callback],
        logger=[tb_logger],
    )
    trainer.fit(net)


if __name__ == "__main__":
    config = OmegaConf.load("config/train_config.yaml")
    main(config)