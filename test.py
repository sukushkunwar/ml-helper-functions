from datasets import load_dataset
from helpers.data_utils import process_and_get_tokenized_dataset, get_data_collator
from transformers import AutoTokenizer
#Configurations
DATASET_NAME = "lhoestq/conll2003"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
MODEL_NAME = ["bert-base-uncased"]
# load the datasets
train_ds = load_dataset(DATASET_NAME, split=TRAIN_SPLIT)
test_ds = load_dataset(DATASET_NAME, split=TEST_SPLIT)
#load tokenizer
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME[0])
#parameters
max_length = 128
padding = "max_length"
truncation = True
#tokenize the dataset
processed_train_ds = process_and_get_tokenized_dataset(dataset=train_ds,
                                             tokenizer=tokenizer,
                                             max_length=max_length,
                                             padding=padding,
                                             truncation=truncation)
processed_test_ds = process_and_get_tokenized_dataset(dataset=test_ds,
                                              tokenizer=tokenizer,
                                              max_length=max_length,
                                             padding=padding,
                                             truncation=truncation)
                                             #get data collator
data_collator = get_data_collator(tokenizer)
from torch.utils.data import DataLoader


train_loader = DataLoader(
    processed_train_ds,
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator
)

train_loader = DataLoader(
    processed_test_ds,
    batch_size=8,
    collate_fn=data_collator
)

batch = next(iter(train_loader))
print(batch)