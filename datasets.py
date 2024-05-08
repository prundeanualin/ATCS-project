from torch.utils.data import Dataset
import pandas as pd
import torch

SCAN_PATH = 'data/SCAN/SCAN_dataset.csv'
SCAN_EXAMPLES_PATH = 'data/SCAN/SCAN_examples.txt'
ANALOGY_TEMPLATE_SIMPLE_FULL = "If {} is like {}, then {} is like"
ANALOGY_TEMPLATE_SIMPLE_INFERENCE = "If {} is like {}, then {} is like"

# active vs passive
# simple wording VS complex wording

def create_analogy_sentence(self, analogy_values, anal):
    if inference_sent:
        return self.inference_analogy.format(*analogy_values)
    else:
        return self.full_analogy.format(*analogy_values)

def get_list_alternatives(alternatives):
    if alternatives == 'nan':
        return []
    else:
        return alternatives.split(', ')


class ScanDataset(Dataset):
    def __init__(self, shot_nr=1, examples_start_idx=0, analogy_sentence_infer=ANALOGY_TEMPLATE_SIMPLE_INFERENCE, analogy_sentence_full=ANALOGY_TEMPLATE_SIMPLE_FULL):
        self.shot_nr = shot_nr
        self.examples_start_idx = examples_start_idx
        self.analogy_sentence_inference = analogy_sentence_infer
        self.analogy_sentence_example = analogy_sentence_full

        with open(SCAN_PATH, 'r', encoding='utf8') as f:
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
                    'detailed': example[1],
                    'simple': self.analogy_sentence_example.format(target, source, targ_word, src_word),
                    'analogy_type': analogy_type
                }
                self.examples[analogy_type].append(ex)

        self.df_remapped_indices = [i for i in range(len(self.df))]
        self.current_examples = {k: self.examples[k][self.examples_start_idx: self.shot_nr] for k in self.examples.keys()}

        # Remapping the indices in df so that those corresponding to the examples_to_consider are removed.
        # This way, the examples will not be returned when iterating through the df
        print(self.current_examples)
        indices_examples = []
        curr_examples = []
        for k in self.current_examples:
            curr_examples.extend(self.current_examples[k])
        for ex in curr_examples:
            idx = self.df[(self.df['source'] == ex['source']) & (self.df['target'] == ex['target']) & (
                    self.df['targ_word'] == ex['targ_word']) & (self.df['targ_word'] == ex['targ_word'])].index
            indices_examples.append(idx.values[0])

        print("Indices of the examples are: ")
        print(indices_examples)

        print("Initial df indicies: ")
        print(self.df_remapped_indices)
        for ex_i in indices_examples:
            self.df_remapped_indices.pop(ex_i)

        print("Final df indicies: ")
        print(self.df_remapped_indices)

        print('Done loading SCAN')

    def __len__(self):
        return int(len(self.df))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.df_remapped_indices[idx]
        analogy_sent = self.df.iloc[idx, :3].values.tolist()
        label = self.df.iloc[idx, 3]
        alternatives = self.df.iloc[idx, 4]
        analogy_type = self.df.iloc[idx, 5]

        return {
            'analogy_sentence': self.analogy_sentence_inference.format(*analogy_sent),
            'examples': self.current_examples,
            'label': label,
            'alternatives': alternatives,
            'analogy_type': analogy_type
        }


if __name__ == '__main__':
    dataset = ScanDataset()
