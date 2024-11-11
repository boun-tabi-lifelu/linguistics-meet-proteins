import pandas as pd
import sentencepiece as spm

vocab_size = 6400

print('Reading Data')
seqs = []
with open("split0.csv") as f: # load_dataset("agemagician/uniref50", split='train') random subset of 15M
    for i, line in enumerate(f):
        line = line.strip()
        seq = line.split(",")[-1]
        seqs.append(seq)

with open("seqs.txt", "w") as f:
    for seq in seqs:
        f.write(f"{seq}\n")

print('Training Start')
spm.SentencePieceTrainer.train(
    input="seqs.txt",
    model_prefix="m",
    vocab_size=vocab_size,
    max_sentence_length=8997,
    train_extremely_large_corpus=True,
)
# Saving model: m.model, Saving vocabs: m.vocab

print("Training Finished")