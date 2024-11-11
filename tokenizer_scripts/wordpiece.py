from tokenizers import Tokenizer, models, normalizers, trainers

vocab_size = 6400

print('Reading Data')
seqs = []
with open("split0.csv") as f: # load_dataset("agemagician/uniref50", split='train') random subset of 15M
    for i, line in enumerate(f):
        line = line.strip()
        seq = line.split(",")[-1]
        seqs.append(seq)

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

trainer = trainers.WordPieceTrainer(
    vocab_size=vocab_size,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
)

print('Training Start')
tokenizer.train_from_iterator(seqs, trainer=trainer)

print("Training Finished")
tokenizer.save(f"wp_{vocab_size}.json")
