import numpy as np
import torch
from dataclasses import dataclass
from typing import List
import pandas as pd

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
    labels: List[int]
    demographics: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            noteid=self.noteid,
            text=self.text.split(" "),
            labels=self.labels,
            demographics=self.demographics
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    noteid: str
    text: List
    labels: List[int]
    demographics: str

    def featurize(self, tokenizer, feat_spec):
        dt = single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.text,
            label_id=self.labels,
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
    label_id: List[int]
    tokens: list
    note_id: str=""


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class PhenotypingTask(Task):
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
            demographic_field: str="gender",
            device="cpu"
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
        if demographic_field is None:
            demographic_field = "gender"
        self.demographic_column = demographic_field
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
            [5.15881033, 17.05098684, 9.20455602, 2.52723767, 6.76441457,
             7.5809226, 5.09383676, 14.01367989, 3.3917567, 1.98436438,
             9.68126521, 4.2853359, 2.31921972, 1.38612893, 3.63961108,
             13.31833007, 6.66143106, 13.38401048, 23.85843715, 35.64440735,
             14.81412104, 11.2012229, 10.46812957, 9.74926543, 24.79318449,
             0.38328712, 0.2727589, 0.10362512], device=device
        ))
        self.fairness_loss_dict = fairness.FAIR_LOSS_DICT["multilabel"]
        self.explicit_subset = None
        self.label_dim = len(self.LABELS)

    def get_explicit_subset(self, subset_size=100, seed=12):
        df = pd.read_pickle(self.train_path)
        val = df[df.fold.isin(self.val_fold_ids)]
        note_ids = val.groupby([self.demographic_column, 'Other upper respiratory disease']).apply(
            lambda df: df.sample(frac=.15, random_state=seed))['note_id'].values.tolist()
        idx = 0
        idxs = []
        for _, row in val.iterrows():
            for _ in row['seqs']:
                if row['note_id'] in note_ids:
                    idxs.append(idx)
                idx += 1
        self.explicit_subset = idxs
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
                    labels=row[self.LABELS].astype(int).tolist(),
                    demographics=row[self.demographic_column]
                )
            )
        return

    def _create_examples(self, df, set_type):
        examples = []
        df.apply(lambda row: self.get_sequences_from_rows(row, examples), axis=1)
        return examples
