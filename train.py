import pytorch_lightning as pl
from torchtext.data import Field, TabularDataset, BucketIterator

from net import Net


def main():

    src = Field(init_token="<sos>", eos_token="<eos>", lower=True)
    tgt = Field(init_token="<sos>", eos_token="<eos>", lower=True)

    dataset = TabularDataset(
        path="data/dataset.tsv", format="tsv", fields=[("src", src), ("tgt", tgt)]
    )
    train, val, test = dataset.split(split_ratio=[0.8, 0.1, 0.1])

    src.build_vocab(train)
    tgt.build_vocab(train)

    src_vocab_size = len(src.vocab)
    tgt_vocab_size = len(tgt.vocab)

    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test),
        batch_size=32,
        sort=False,
    )

    net = Net(train_iter, val_iter, test_iter, src_vocab_size, tgt_vocab_size)
    trainer = pl.Trainer()
    trainer.fit(net)


if __name__ == "__main__":
    main()