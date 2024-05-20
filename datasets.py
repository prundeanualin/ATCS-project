from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm

from get_datasets import *
from prompt_templates.analogy import *


def get_list_alternatives(alternatives):
    if alternatives == 'nan':
        return []
    else:
        return alternatives.split(', ')


class ScanDataset(Dataset):
    def __init__(self,
                 shuffle=False,
                 analogy_sentence_infer=ANALOGY_TEMPLATE_SIMPLE_INFERENCE,
                 analogy_sentence_full=ANALOGY_TEMPLATE_SIMPLE_FULL,
                 examples_file=SCAN_EXAMPLES_FILEPATH,
                 examples_start_idx=0,
                 examples_shot_nr=1):
        """
            - examples_start_idx: needs to be within the range of available examples per analogy type.
                For the current SCAN setup, this means 25 examples per analogy type.
            - shot_nr: number of examples to be considered from each analogy type
        """

        self.analogy_sentence_inference = analogy_sentence_infer
        self.analogy_sentence_example = analogy_sentence_full
        self.shuffle = shuffle

        get_datasets_if_not_present()

        # Load the dataset from file
        with open(SCAN_DATASET_FILEPATH, 'r', encoding='utf8') as f:
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
        self.examples_file = examples_file
        self.examples_shot_nr = examples_shot_nr
        self.examples_start_idx = examples_start_idx

        self.examples, self.current_examples, self.df_remapped_indices = None, None, None

        self.read_examples(self.examples_file)
        self.set_examples_to_use_few_shot(self.examples_start_idx, self.examples_shot_nr)

    def read_examples(self, examples_file):
        self.examples_file = examples_file
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
                    'detailed_cot': example[1],
                    'simple': self.analogy_sentence_example.format(target, source, targ_word, src_word),
                    'analogy_type': analogy_type
                }
                self.examples[analogy_type].append(ex)

    def set_examples_to_use_few_shot(self, examples_start_idx, nr_examples_few_shot):
        self.examples_start_idx = examples_start_idx
        self.examples_shot_nr = nr_examples_few_shot

        # Get X (=nr_examples) examples starting from the start_idx, for each analogy type
        self.current_examples = []
        for k in self.examples.keys():
            self.current_examples.extend(self.examples[k][self.examples_start_idx: self.examples_start_idx + self.examples_shot_nr])

        # Remapping the indices in df so that those corresponding to the examples_to_consider are removed.
        # This way, the examples will not be returned when iterating through the df
        self.df_remapped_indices = [i for i in range(len(self.df))]
        indices_examples = []
        for ex in self.current_examples:
            idx = self.df[(self.df['source'] == ex['source']) & (self.df['target'] == ex['target']) & (
                    self.df['targ_word'] == ex['targ_word']) & (self.df['src_word'] == ex['src_word'])].index
            indices_examples.append(idx.values[0])
        self.df_remapped_indices = [i for i in range(len(self.df)) if i not in indices_examples]

    def __len__(self):
        return int(len(self.df_remapped_indices))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx_excluding_examples = self.df_remapped_indices[idx]
        item = self.df.iloc[idx_excluding_examples]

        return {
            'inference': self.analogy_sentence_inference.format(item['target'], item['source'], item['targ_word']),
            'label': item['src_word'],
            'alternatives': item['alternatives'],
            'analogy_type': item['analogy_type']
        }


if __name__ == '__main__':
    dataset = ScanDataset(
        shuffle=False,
        analogy_sentence_infer=ANALOGY_TEMPLATE_SIMPLE_INFERENCE,
        analogy_sentence_full=ANALOGY_TEMPLATE_SIMPLE_FULL,
        examples_file=SCAN_EXAMPLES_FILEPATH.format(EXAMPLE_CATEGORIES[0]),
        examples_start_idx=5,
        examples_shot_nr=5
    )
    print("Examples are: ")
    for ex in dataset.current_examples:
        print(ex['simple'])
    print("\n\n")

    for i, sample in tqdm(enumerate(dataset)):
        print(sample['inference'])