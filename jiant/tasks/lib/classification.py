import numpy as np
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
from torch.nn import CrossEntropyLoss
from jiant.ext.fairness import DF_training as fairness


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    label: int
    demographics: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            text=tokenizer.tokenize(self.text),
            label_id=self.label,
            demographics=self.demographics
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text: List
    label_id: int
    demographics: str

    def featurize(self, tokenizer, feat_spec):
        return single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.text,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class ClassificationTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION

    def __init__(
            self, name: str, 
            path_dict: dict, 
            num_labels: int=None, 
            labels: List[str]=None, 
            demographics: List[str]=None, 
            train_size: int=0, 
            device=None,
            demographic_field=None
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
        self.loss = CrossEntropyLoss()
        self.fairness_loss_dict = fairness.FAIR_LOSS_DICT["multiclass"]
        self.label_dim = 1
        if demographic_field is not None:
            self.demographic_column = demographic_field
            if demographic_field == "age":
                self.demographics = ['U35', 'O45']
            else:
                self.demographics = ['M', 'F']
        else:
            self.demographic_column = ["gender", "age"]


    def get_train_examples(self):
        return self._create_examples(lines=read_jsonl(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_jsonl(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_jsonl(self.test_path), set_type="test")

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    text=line["text"],
                    label=self.LABEL_TO_ID[line['label']],
                    demographics=f"{line['gender']}-{line['age']}"
                )
            )
        return examples
