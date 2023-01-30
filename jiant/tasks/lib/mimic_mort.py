import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import List

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    GlueMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import single_sentence_featurize, labels_to_bimap
from jiant.utils.python.io import read_jsonl
from jiant.ext.fairness import DF_training as fairness


@dataclass
class Example(BaseExample):
    guid: str
    noteid: str
    text: str
    label: int
    demographics: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            noteid=self.noteid,
            text=self.text.split(" "),
            label_id=self.label,
            demographics=self.demographics
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    noteid: str
    text: List
    label_id: int
    demographics: str

    def featurize(self, tokenizer, feat_spec):
        dt = single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.text,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )
        dt.note_id = self.noteid
        return dt


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list
    note_id: str=""


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class InHospitalMortalityTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION

    def __init__(self, 
        name: str, 
        path_dict: dict, 
        num_labels: int=None, 
        labels: List[str]=None, 
        demographics: List[str]=None, 
        train_size: int=0,
        val_fold_ids: List[str]=["1", "2"],
        demographic_column: str="gender",
        device='cpu'
        ):
        super().__init__(name, path_dict)
        if num_labels is None and labels is None:
            raise TypeError("num_labels or labels must be passed!")
        if num_labels is not None and num_labels > 0:
            self.LABELS = [str(i) for i in range(num_labels)]
        elif labels is not None and len(labels) > 0:
            self.LABELS = labels
        else:
            raise TypeError("wrong format for num_labels or labels")
        self.LABEL_TO_ID, self.ID_TO_LABEL = labels_to_bimap(self.LABELS)
        if demographics is not None:
            self.demographics = demographics
        if train_size != 0:
            self.train_size = train_size
        self.val_fold_ids = val_fold_ids
        self.demographic_column = demographic_column
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.5], device=device))
        self.fairness_loss_dict = fairness.FAIR_LOSS_DICT["binary"]
        self.explicit_subset = None
    

    def get_explicit_subset(self, subset_size=100, seed=12):
        df = pd.read_pickle(self.train_path)
        val = df[df.fold.isin(self.val_fold_ids)]
        note_ids = val.groupby([self.demographic_column, 'inhosp_mort']).apply(
            lambda df: df.sample(n=subset_size, random_state=seed))['note_id'].values.tolist()
        idx = 0
        idxs = []
        for _, row in val.iterrows():
            for _ in row['seqs']:
                if row['note_id'] in note_ids:
                    idxs.append(idx)
                idx += 1
        self.explicit_subset =  idxs
        self.val_note_id_subset = val['note_id'].isin(note_ids)



    def get_train_examples(self):
        df = pd.read_pickle(self.train_path)
        train = df[~df.fold.isin(["test"] + self.val_fold_ids)]
        return self._create_examples(df=train, set_type="train")

    def get_val_examples(self):
        df = pd.read_pickle(self.train_path)
        val = df[df.fold.isin(self.val_fold_ids)]
        return self._create_examples(df=val, set_type="val")

    def get_test_examples(self):
        df = pd.read_pickle(self.train_path)
        test = df[df.fold == "test"]
        return self._create_examples(df=test, set_type="test")
    
    def get_val_labels(self, cache):
        val_labels = {}
        for datum in cache.iter_all():
            if datum["data_row"].note_id in val_labels:
                assert(val_labels[datum["data_row"].note_id] == datum["data_row"].label_id)
            else:
                val_labels[datum["data_row"].note_id] = datum["data_row"].label_id
        return np.array(list(val_labels.values()))
    
    def get_sequences_from_rows(self, row, examples):
        for i, seq in enumerate(row['seqs']):
            examples.append(
                Example(
                    guid=f"{row['note_id']}_{i}",
                    noteid=row["note_id"],
                    text=seq,
                    label=int(row["inhosp_mort"]),
                    demographics=row[self.demographic_column]
                )
            )
        return 

    def _create_examples(self, df, set_type):
        examples = []
        df.apply(lambda row: self.get_sequences_from_rows(row, examples), axis=1)
        return examples
