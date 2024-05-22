import pandas as pd
import torch
import random

from get_datasets import *
from prompt_processing.templates import *


def get_list_alternatives(alternatives):
    if alternatives == 'nan':
        return []
    else:
        return alternatives.split(', ')


def is_example_equal_to_sample(sample, ex):
    return ((sample['source'] == ex['source']) and
            (sample['target'] == ex['target']) and
            (sample['targ_word'] == ex['targ_word']) and
            (sample['src_word'] == ex['src_word']))


class ScanDataloader:
    def __init__(self,
                 shuffle=False,
                 analogy_sentence_infer=ANALOGY_TEMPLATE_SIMPLE_INFERENCE,
                 analogy_sentence_full=ANALOGY_TEMPLATE_SIMPLE_FULL,
                 dataset_file=SCAN_DATASET_FILEPATH,
                 examples_file=SCAN_EXAMPLES_FILEPATH,
                 examples_shot_nr=1):
        """
            - examples_shot_nr: number of examples to be considered from each analogy type
        """
        self.analogy_sentence_inference = analogy_sentence_infer
        self.analogy_sentence_example = analogy_sentence_full
        self.shuffle = shuffle
        self.length = 0

        get_datasets_if_not_present()

        # Load the dataset from file
        with open(dataset_file, 'r', encoding='utf8') as f:
            self.df = pd.read_csv(f, sep=',', index_col=False)

        # Transform alternatives into list of strings. If none is provided, then use an empty list
        self.df['alternatives'] = self.df['alternatives'].astype(str).map(get_list_alternatives)

        # there were still some rows with same first three words but different fourth word. This should not
        # be the case since then the second instance should be part of the first's alternatives
        last_doubled = self.df[self.df.duplicated(subset=['source', 'target', 'targ_word'], keep='last')]
        self.df.drop(last_doubled.index, axis=0, inplace=True)
        self.df.reset_index(inplace=True)

        for _, row in last_doubled.iterrows():
            first = self.df[(self.df['source'] == row['source']) & (self.df['target'] == row['target']) & (
                    self.df['targ_word'] == row['targ_word'])].index

            self.df.at[first.values[0], 'alternatives'] = self.df.at[first.values[0], 'alternatives'] + row[
                'alternatives'] + [row['src_word']]

        # Shuffle the dataset, if necessary
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        # Process examples from file
        self.examples_shot_nr = examples_shot_nr
        self.examples = None
        self.read_examples(examples_file)

    def read_examples(self, examples_file):
        self.examples = {
           analogy_type: [] for analogy_type in self.df['analogy_type'].unique()
        }
        # Load the examples from file
        with open(examples_file, 'r', encoding='utf8') as f:
            lines = [line.rstrip() for line in f]
            total_nr_examples = round(len(lines) / 3)
            examples = [lines[idx * 3: (idx * 3) + 2] for idx in range(0, total_nr_examples)]
            for example in examples:
                target, source, targ_word, src_word, alternatives, analogy_type = tuple(example[0].split(','))
                ex = {
                    'target': target,
                    'source': source,
                    'targ_word': targ_word,
                    'src_word': src_word,
                    'analogy_incomplete': self.analogy_sentence_inference.format(target, source, targ_word),
                    'analogy_complete': self.analogy_sentence_example.format(target, source, targ_word, src_word),
                    'analogy_detailed_cot': example[1],
                    'analogy_type': analogy_type
                }
                self.examples[analogy_type].append(ex)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = self.df.iloc[idx]

        # Load shot_nr random examples from the same analogy category,
        # while making sure they are different from the sample
        examples = []
        for i in range(self.examples_shot_nr):
            ex = random.choice(self.examples[item['analogy_type']])
            while is_example_equal_to_sample(item, ex):
                ex = random.choice(self.examples[item['analogy_type']])
            examples.append(ex)

        return {
            'inference': self.analogy_sentence_inference.format(item['target'], item['source'], item['targ_word']),
            'label': item['src_word'],
            'alternatives': item['alternatives'],
            'analogy_type': item['analogy_type'],
            'examples': examples
        }


if __name__ == '__main__':
    dataset = ScanDataloader(
        shuffle=False,
        analogy_sentence_infer=ANALOGY_TEMPLATE_SIMPLE_INFERENCE,
        analogy_sentence_full=ANALOGY_TEMPLATE_SIMPLE_FULL,
        examples_file=SCAN_EXAMPLES_FILEPATH.format('baseline'),
        examples_shot_nr=8
    )
    for i, sample in enumerate(dataset):
        print(sample['inference'] + "  -- " + sample['label'])
        for ex in sample['examples']:
            print(ex['analogy_complete'])
        print("----\n")

    # Test is_example_equal_to_sample method
    print("Is example equal to sample: ")
    initial_elem = dataset.df.iloc[5]
    elem = dataset.df.iloc[11]
    assert not is_example_equal_to_sample(initial_elem, elem)
    elem['target'] = initial_elem['target']
    elem['source'] = initial_elem['source']
    elem['targ_word'] = initial_elem['targ_word']
    elem['src_word'] = initial_elem['src_word']
    assert is_example_equal_to_sample(dataset.df.iloc[5], dataset.df.iloc[5])

