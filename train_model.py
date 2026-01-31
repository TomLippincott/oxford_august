import csv
import gzip
import logging
from glob import glob
from copy import deepcopy
import random
import os.path
import argparse
import torch
from torch.nn import Module, Embedding, LSTM, Linear
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.utils.data import DataLoader


logger = logging.getLogger("train_model")


def collate(items):
    words = pad_sequence([x for x, _, _ in items], batch_first=True)
    stresses = pad_sequence([x for _, x, _ in items], batch_first=True)
    lengths = torch.tensor([x for _, _, x in items])
    return (words, stresses, lengths)


class Model(Module):
    def __init__(self, embedding_size, word_vocab, hidden_size, stress_vocab, bidirectional):
        super(Model, self).__init__()
        self.word_vocab = word_vocab
        self.stress_vocab = stress_vocab
        self.embedding = Embedding(len(self.word_vocab), embedding_size)
        self.rnn = LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional, dropout=0.0)
        self.head = Linear(hidden_size * (2 if bidirectional else 1), len(self.stress_vocab))

    def forward(self, words):
        out, _ = self.rnn(self.embedding(words))
        return self.head(out)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="full_dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_proportion", type=float, default=0.8)
    parser.add_argument("--dev_proportion", type=float, default=0.1)
    parser.add_argument("--test_proportion", type=float, default=0.1)
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--use_dev_loss", action="store_true", default=False)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--unidirectional", action="store_true", default=False)
    parser.add_argument("--device")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    if args.seed != None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.set_default_device(device)
    logger.info("Using device = %s", torch.get_default_device())
        
    stress_vocab = {}
    word_vocab = {}
    data = []
    for fname in glob(os.path.join(args.data_path, "*csv")):
        if os.path.basename(fname) == "combined.csv":
            continue
        with open(fname, "rt") as ifd:
            for row in csv.DictReader(ifd):
                o_words = row["OXFORD_TEXT"].split()
                r_words = row["RIVERSIDE_TEXT"].split()
                o_stresses = row["OXFORD_SCANSION"].split()
                r_stresses = row["RIVERSIDE_SCANSION"].split()
                words = o_words
                stresses = o_stresses
                if len(words) == len(stresses):
                    data.append(
                        (
                            torch.tensor([word_vocab.setdefault(w, len(word_vocab)) for w in words]),
                            torch.tensor([stress_vocab.setdefault(s, len(stress_vocab)) for s in stresses]),
                            len(words)
                        )
                    )

    random.shuffle(data)

    train_count = int(args.train_proportion * len(data))
    dev_count = int(args.dev_proportion * len(data))
    test_count = int(args.dev_proportion * len(data))
    
    train = data[:train_count]
    dev = data[train_count:train_count + dev_count]
    test = data[train_count + dev_count:]

    logger.info(
        "Split data into %d/%d/%d train/dev/test lines, word vocabulary size=%d, stress pattern vocabulary size=%d",
        len(train),
        len(dev),
        len(test),
        len(word_vocab),
        len(stress_vocab)
    )
    
    train_dl = DataLoader(train, batch_size=args.batch_size, collate_fn=collate, shuffle=True, generator=torch.Generator(device=device))
    dev_dl = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate)
    test_dl = DataLoader(test, batch_size=args.batch_size, collate_fn=collate)

    model = Model(args.embedding_size, word_vocab, args.hidden_size, stress_vocab, not args.unidirectional)
    
    opt = AdamW(model.parameters(), lr=args.learning_rate)

    best_dev_score = None
    best_state_dict = None
    for epoch in range(args.max_epochs):
        train_losses = []
        dev_losses = []
        dev_guesses = []
        dev_golds = []
        model.train()
        for words, stresses, lengths in train_dl:
            out = model(words)
            ces = cross_entropy(torch.transpose(out, 1, 2), stresses, reduction="none")
            uces = unpad_sequence(ces, lengths, batch_first=True)
            loss = sum([x.sum() for x in uces])
            loss.backward()
            clip_grad_norm_(model.parameters(), 3)
            opt.step()
            opt.zero_grad()
            train_losses.append(loss.detach().item())
        train_loss = sum(train_losses) / (len(train_losses) * args.batch_size)
        model.eval()
        for words, stresses, lengths in dev_dl:
            out = model(words)
            dev_guesses.append(torch.cat(unpad_sequence(out.argmax(dim=2).detach(), lengths, batch_first=True)))
            dev_golds.append(torch.cat(unpad_sequence(stresses.detach(), lengths, batch_first=True)))            
            kls = cross_entropy(torch.transpose(out, 1, 2), stresses, reduction="none")
            loss = sum([x.sum() for x in unpad_sequence(kls, lengths, batch_first=True)])
            dev_losses.append(loss.detach().item())
        dev_loss = sum(dev_losses) / (len(dev_losses) * args.batch_size)
        dev_guesses = torch.cat(dev_guesses)
        dev_golds = torch.cat(dev_golds)
        dev_acc = (dev_guesses == dev_golds).sum() / dev_guesses.shape[0]
        logger.info(
            "Epoch %d average instance loss (train/dev): %f/%f\n  Dev accuracy: %f",
            epoch + 1,
            train_loss,
            dev_loss,
            dev_acc
        )        
        dev_score = -dev_loss if args.use_dev_loss else dev_acc
        if not best_dev_score or dev_score > best_dev_score:
            logger.info("Saving new best model")
            best_dev_score = dev_score
            best_state_dict = deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    model.eval()
    test_losses = []
    test_guesses = []
    test_golds = []    
    for words, stresses, lengths in test_dl:
        out = model(words)
        test_guesses.append(torch.cat(unpad_sequence(out.argmax(dim=2).detach(), lengths, batch_first=True)))
        test_golds.append(torch.cat(unpad_sequence(stresses.detach(), lengths, batch_first=True)))            
        kls = cross_entropy(torch.transpose(out, 1, 2), stresses, reduction="none")
        loss = sum([x.sum() for x in torch.nn.utils.rnn.unpad_sequence(kls, lengths, batch_first=True)])
        test_losses.append(loss.detach().item())
    test_guesses = torch.cat(test_guesses)
    test_golds = torch.cat(test_golds)
    test_acc = (test_guesses == test_golds).sum() / test_guesses.shape[0]        
    logger.info("Final average instance loss (test): %f\n  Test accuracy: %f", sum(test_losses) / (len(test_losses) * args.batch_size), test_acc)
    
    with open(args.output, "wb") as ofd:
        torch.save(model, ofd)
