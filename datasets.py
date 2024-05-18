from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm
from itertools import combinations
import random
import pickle

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
            self.current_examples.extend(self.examples[k][self.examples_start_idx: self.examples_shot_nr])

        # Remapping the indices in df so that those corresponding to the examples_to_consider are removed.
        # This way, the examples will not be returned when iterating through the df
        self.df_remapped_indices = [i for i in range(len(self.df))]
        indices_examples = []
        for ex in self.current_examples:
            idx = self.df[(self.df['source'] == ex['source']) & (self.df['target'] == ex['target']) & (
                    self.df['targ_word'] == ex['targ_word']) & (self.df['src_word'] == ex['src_word'])].index
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
            'label': label,
            'alternatives': alternatives,
            'analogy_type': analogy_type
        }

class BATSDataloader_0shot:
    def __init__(self, dataFolder, fileName, numberOfAnalogy = False, cot=False, shuffle=False, promptType="by-relation", promptFormat=" If {} is like {}, {} is like ..."):
        self.dataFolder = dataFolder
        self.fileName = fileName
        self.promptFormat = promptFormat
        start = fileName.find('[') + 1
        end = fileName.find(']')
        self.analogyType = fileName[start:end]
        self.numberOfAnalogy = numberOfAnalogy
        self.promptType = promptType # promptType: by-relation, by-target-word
        self.COT = cot
        self.promptFormat = promptFormat
        self.shuffle = shuffle
        self.load_pairs()
        self.build_prompt()

    def load_pairs(self):
        '''
        Output: pairs = [[target, source, alternatives, analogyType], [target, source, alternatives, analogyType], ...]
        '''
        with open(f'{self.dataFolder}/{self.fileName}.txt', 'r') as f:
            lines = f.readlines()
        self.pairs = []
        for line in lines:
            target, values = line.strip().split('\t')
            values = values.split('/')
            # remove value that have underscore
            values = [value for value in values if '_' not in value]
            # only select the first element in value list as label/attribute, the rest are alternatives
            source = values[0]
            alternatives = [value for value in values[1:] if value]
            self.pairs.append([target, source, alternatives, self.analogyType])

        # # save as csv
        # with open(f'{self.dataFolder}/{self.fileName}_pairs.csv', 'w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['target', 'source', 'alternatives', 'analogy_type'])
        #     writer.writerows(self.pairs)

        # save as pickle
        if not os.path.exists(f'{self.dataFolder}/pairs'):
            os.makedirs(f'{self.dataFolder}/pairs')
        with open(f'{self.dataFolder}/pairs/{self.fileName}.pkl', 'wb') as f:
            pickle.dump(self.pairs, f)


    def build_prompt(self):
        '''
        Output:
        '''
        self.prompt = []
        if self.promptType == "by-relation":
            # combination of 2 pairs and generate prompt
            for pair in combinations(self.pairs, 2):
                # check if same analogy type
                if pair[0][3] != pair[1][3]:
                    continue
                inference = self.promptFormat.format(pair[0][0], pair[0][1], pair[1][0])
                if self.COT:
                    inference = self.COT + inference
                self.prompt.append({'inference': inference, 'label': pair[1][1], 'alternatives': pair[1][2], 'analogy_type': pair[0][3]})
        elif self.promptType == "by-target-word":
            for pair in combinations(self.pairs, 2):
                # check if same analogy type
                if pair[0][3] != pair[1][3]:
                    continue
                inference = self.promptFormat.format(pair[0][0], pair[1][0], pair[0][1])
                if self.COT:
                    inference = self.COT + inference
                self.prompt.append({'inference': inference, 'label': pair[1][1], 'alternatives': pair[1][2], 'analogy_type': pair[0][3]})
        else:
            print("Invalid prompt type. Either by-relation or by-target-word")

        fname = self.get_file_name()
        
        # # save as csv
        # with open(f'{self.dataFolder}/{fname}.csv', 'w') as f:
        #     writer = csv.DictWriter(f, fieldnames=self.prompt[0].keys())
        #     writer.writeheader()
        #     writer.writerows(self.prompt)

        # save as pickle
        if not os.path.exists(f'{self.dataFolder}/prompt'):
            os.makedirs(f'{self.dataFolder}/prompt')
        with open(f'{self.dataFolder}/prompt/{fname}.pkl', 'wb') as f:
            pickle.dump(self.prompt, f)

    def get_file_name(self):
        prompt_condition = self.promptFormat.split(",")[0]
        return f"0shot_{self.fileName}_{self.COT if self.COT else 'no_COT'}_{self.promptType}_{prompt_condition}"

    def __call__(self):
        if self.shuffle:
            random.shuffle(self.prompt)
        # return only the number of analogy specified (if does not exceed the total number of analogy)
        if self.numberOfAnalogy and self.numberOfAnalogy < len(self.pairs):
            return self.prompt[:self.numberOfAnalogy]
        return self.prompt

if __name__ == '__main__':
    # dataset = ScanDataset(
    #     shuffle=False,
    #     analogy_sentence_infer=ANALOGY_TEMPLATE_SIMPLE_INFERENCE,
    #     analogy_sentence_full=ANALOGY_TEMPLATE_SIMPLE_FULL,
    #     examples_file=SCAN_EXAMPLES_FILEPATH.format(EXAMPLE_CATEGORIES[0]),
    #     examples_start_idx=0,
    #     examples_shot_nr=1
    # )
    # for i, sample in tqdm(enumerate(dataset)):
    #     print(sample['inference'])

    BATS_dataset = BATSDataloader_0shot(
        BATS_FOLDER, 
        BATS_FILENAME, 
        numberOfAnalogy = 2, 
        cot = COT_TEMPLATE,
        shuffle = False,
        promptType='by-relation', 
        promptFormat = ANALOGY_TEMPLATE_SIMPLE_INFERENCE
    )

    # print(BATS_dataset.get_file_name()) # 0shot_L01 [hypernyms - animals] sample_Thinking step by step. _by-relation_If {} is like {}

    for id, sample in enumerate(BATS_dataset()):
        print(sample['inference'])
        print(sample['label'])
        print(sample['alternatives'])
        print(sample['analogy_type'])
        print()