from datasets import load_dataset
import random
import pandas as pd
from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

vocab_size = 6400

print('Reading Data')
dataset_train = load_dataset("wikitext", "wikitext-103-raw-v1", split='train')

print('Preprocessing Data')
df_ds_train = pd.DataFrame(dataset_train)
df_ds_train['text'] = df_ds_train['text'].str.strip().str.strip('= \n')
df_ds_train = df_ds_train[df_ds_train['text'] != '']
print(len(df_ds_train))

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=2,
    initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet())[:94], # string.ascii_letters+string.digits+string.punctuation
    limit_alphabet=94,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
)

print('Training Start')
tokenizer.train_from_iterator(iterator=list(df_ds_train['text']), trainer=trainer, length=len(df_ds_train))

print("Training Finished")
tokenizer.save(f"bpe_en_{vocab_size}.json")

