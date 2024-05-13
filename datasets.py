from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from tqdm import tqdm

from prompt_templates.analogy import ANALOGY_TEMPLATE_SIMPLE_INFERENCE, ANALOGY_TEMPLATE_SIMPLE_FULL

SCAN_DATASET_PATH = 'data/SCAN/SCAN_dataset.csv'
SCAN_EXAMPLES_PATH = 'data/SCAN/SCAN_examples.txt'

def get_list_alternatives(alternatives):
    if alternatives == 'nan':
        return []
    else:
        return alternatives.split(', ')


class ScanDataset(Dataset):
    def __init__(self,
                 shot_nr=1,
                 examples_start_idx=0,
                 analogy_sentence_infer=ANALOGY_TEMPLATE_SIMPLE_INFERENCE,
                 analogy_sentence_full=ANALOGY_TEMPLATE_SIMPLE_FULL):
        """
          - examples_start_idx: needs to be within the range of available examples per analogy type.
        """

        self.shot_nr = shot_nr
        self.examples_start_idx = examples_start_idx
        self.analogy_sentence_inference = analogy_sentence_infer
        self.analogy_sentence_example = analogy_sentence_full

        with open(SCAN_DATASET_PATH, 'r', encoding='utf8') as f:
            self.df = pd.read_csv(f, sep=',', index_col=False)

        # Transform alternatives into list of strings. If none is provided, then use an empty list
        self.df['alternatives'] = self.df['alternatives'].astype(str).map(get_list_alternatives)

        # there were still some rows with same first three words but different fourth word. This should not
        # be the case since then the second instance should be part of the first's alternatives
        last_doubled = self.df[self.df.duplicated(subset=['source', 'target', 'targ_word'], keep='last')]
        self.df.drop(last_doubled.index, axis=0, inplace=True)

        for _, row in last_doubled.iterrows():
            first = self.df[(self.df['source'] == row['source']) & (self.df['target'] == row['target']) & (
                    self.df['targ_word'] == row['targ_word'])].index

            self.df.at[first.values[0], 'alternatives'] = self.df.at[first.values[0], 'alternatives'] + row[
                'alternatives'] + [row['src_word']]
        self.examples = {
            'science': [],
            'metaphor': []
        }

        with open(SCAN_EXAMPLES_PATH, 'r', encoding='utf8') as f:
            lines = [line.rstrip() for line in f]
            nr_examples = round(len(lines) / 3)
            examples = [lines[idx * 3: (idx * 3) + 2] for idx in range(0, nr_examples)]
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

        self.current_examples = None
        self.df_remapped_indices = None
        self.set_current_examples_and_exclude_from_dataset(self.examples_start_idx, self.shot_nr)

    def set_current_examples_and_exclude_from_dataset(self, examples_start_idx, nr_examples):
        # Get X (=nr_examples) examples starting from the start_idx, for each analogy type
        self.current_examples = []
        for k in self.examples.keys():
            self.current_examples.extend(self.examples[k][self.examples_start_idx: self.shot_nr])

        # Remapping the indices in df so that those corresponding to the examples_to_consider are removed.
        # This way, the examples will not be returned when iterating through the df
        self.df_remapped_indices = [i for i in range(len(self.df))]
        indices_examples = []
        for ex in self.current_examples:
            idx = self.df[(self.df['source'] == ex['source']) & (self.df['target'] == ex['target']) & (
                    self.df['targ_word'] == ex['targ_word']) & (self.df['targ_word'] == ex['targ_word'])].index
            indices_examples.append(idx.values[0])
        for ex_i in indices_examples:
            self.df_remapped_indices.pop(ex_i)

    def __len__(self):
        return int(len(self.df_remapped_indices))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.df_remapped_indices[idx]
        analogy_sent = self.df.iloc[idx, :3].values.tolist()
        label = self.df.iloc[idx, 3]
        alternatives = self.df.iloc[idx, 4]
        analogy_type = self.df.iloc[idx, 5]

        return {
            'inference': self.analogy_sentence_inference.format(*analogy_sent),
            'examples': self.current_examples,
            'label': label,
            'alternatives': alternatives,
            'analogy_type': analogy_type
        }


if __name__ == '__main__':
    dataset = ScanDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, sample in tqdm(enumerate(dataloader)):
        print(sample)