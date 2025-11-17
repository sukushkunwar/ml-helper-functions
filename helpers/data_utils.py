"""This helper file handles data ingestion and preprocesing for the dataset

It will contain the following functions:
1. `load_hf_dataset(dataset_name, split)` : Load from Hugging Face datasets 
2. `process_dataset(dataset, tokenizer, max_length, padding, truncation)` : Tokenize and prepare features
3. `get_data_collator()`: Return a data collator for dynamic padding
4. `get_tokenized_datasets(dataset_name, split, tokenizer, max_length, padding, truncation)`: Get tokenized datasets
5. `get_train_test_validation_datasets(dataset_name, split, tokenizer, max_length, padding, truncation)`: Get train and test datasets
"""

#importing the necessary libraries
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from typing import Dict, List

#function to load the dataset from the Hugging Face datasets
def load_hf_dataset(dataset_name: str, split: str) -> Dataset:
    """Load dataset from Hugging Face datasets
    
    Args:
        dataset_name: Name of the dataset to load
        split: Split of the dataset to load

    Returns:
        Dataset: Dataset from Hugging Face datasets
    
    Note:
        For datasets that previously used loading scripts (like 'conll2003'), 
        use the new Parquet-based versions instead:
        - Old: 'conll2003' → New: 'conll2003' with 'conll2003' config
        - Or use community versions like: 'eriktks/conll2003'
    
    Example:
        # Using the new format for CoNLL-2003
        train_ds = load_hf_dataset("eriktks/conll2003", "train")
        
        # Or for other datasets
        train_ds = load_hf_dataset("nlpaueb/finer-139", "train")
        test_ds = load_hf_dataset("nlpaueb/finer-139", "test")
        val_ds  = load_hf_dataset("nlpaueb/finer-139", "validation")
    """
    return load_dataset(dataset_name, split=split)


#function to process the dataset
def process_and_get_tokenized_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int, padding: str, truncation: bool, label_all_tokens: bool = False) -> Dataset:
    """Process dataset and get tokenized dataset
    
    Args:
        dataset: Dataset to process
        tokenizer: Tokenizer to use
        max_length: Maximum length of the tokens
        padding: Padding strategy
        truncation: Whether to truncate the tokens
        label_all_tokens: Whether to label all tokens or only the first token of a word

    Returns:
        Dataset: Tokenized dataset
    
    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> max_length = 128
        >>> padding = "max_length"
        >>> truncation = True
        >>> train_ds = process_dataset(train_ds, tokenizer, max_length, padding, truncation)
        >>> test_ds = process_dataset(test_ds, tokenizer, max_length, padding, truncation)
        >>> val_ds = process_dataset(val_ds, tokenizer, max_length, padding, truncation)
    """
    
    def tokenize_and_align_labels(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """Tokenize and align the labels
        
        Args:
            examples: Examples to process

        Returns:
            Dict[str, List[int]]: Tokenized and aligned labels
        """
        tokenized_inputs = tokenizer(examples["tokens"], padding=padding, truncation=truncation, max_length=max_length, is_split_into_words=True)
        all_labels = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(labels[word_idx])
                else:
                    label_ids.append(labels[word_idx] if label_all_tokens else -100)

                previous_word_idx = word_idx

            all_labels.append(label_ids)

        #  Keep ONLY what the model needs
        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": all_labels
    }

    return dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset.column_names)

#function to get the data collator
def get_data_collator(tokenizer: AutoTokenizer) -> DataCollatorForTokenClassification:
    """Get a data collator for token classification
    
    Args:
        tokenizer: Tokenizer to use

    Returns:
        DataCollatorForTokenClassification: Data collator for token classification
    
    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> data_collator = get_data_collator(tokenizer)
    """
    return DataCollatorForTokenClassification(tokenizer=tokenizer)


#function to get the train and test validation datasets
def get_train_test_datasets(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int, padding: str, truncation: bool) -> Dataset:
    """Get train and test validation datasets
    
    Args:
        dataset: Dataset to get train and test validation datasets
        tokenizer: Tokenizer to use
        max_length: Maximum length of the tokens
        padding: Padding strategy
        truncation: Whether to truncate the tokens

    Returns:
        Dataset: Train and test validation datasets
    
    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> max_length = 128
        >>> padding = "max_length"
        >>> truncation = True
    """
    return dataset.train_test_split(test_size=0.1)

