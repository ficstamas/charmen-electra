from typing import Iterator

from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
import gzip
from typing import List
import tqdm
from utils.pretraining.task import mlm, sop

import torch
from datasets import IterableDataset as DIterableDataset
from datasets.iterable_dataset import ExamplesIterable, BufferShuffledExamplesIterable, \
    RandomlyCyclingMultiSourcesExamplesIterable


def generator_function(file, number_of_sentences=2, segment_len_target=2):
    segments = []
    segment_len = 0
    text = ""
    with gzip.open(file, mode="r") as g:
        while True:
            row = g.readline().decode()
            if len(row) == 0:
                return
            if not row.startswith('# text ='):
                continue
            row = row.rstrip('\n')
            if segment_len < segment_len_target:
                text = text + " " + row[9:]
                segment_len += 1
            if len(segments) == number_of_sentences and segment_len == segment_len_target:
                if len(segments) == 1:
                    yield segments[0]
                else:
                    yield 'sentences', {'sentence1': segments[0], 'sentence2': segments[1]}
                segments = []
                text = ""
                segment_len = 0
            if segment_len == segment_len_target:
                segments.append(text[1:])
                text = ""
                segment_len = 0


class ChainDataset(IterableDataset):
    r"""Dataset for chaining multiple :class:`IterableDataset` s.
    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.
    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """

    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, datasets) -> None:
        super(ChainDataset, self).__init__()
        self.datasets = datasets
        self._active = []

    def __iter__(self):
        if len(self._active) == 0:
            self._active = [iter(d) for d in self.datasets]

        while True:
            if len(self._active) == 0:
                return
            idx = np.random.randint(0, len(self._active))
            stream = self._active[idx]
            try:
                line = self.generator_wrapper(stream)
                yield line
            except StopIteration:
                self._active.remove(stream)
                continue

    def __len__(self):
        return len(self.datasets)

    @staticmethod
    def generator_wrapper(stream):
        return stream.__next__()


class Webcorpus:
    def __init__(self, files: List[str]):
        self._files = files

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def get_iterator(self):
        files = [ExamplesIterable(generator_function, {'file': f}) for f in self._files]
        cycling = RandomlyCyclingMultiSourcesExamplesIterable(files, seed=0)
        buffered = BufferShuffledExamplesIterable(cycling, buffer_size=2000, seed=0)
        return DIterableDataset(buffered)


class WebcorpusInMemory(Dataset):
    def __init__(self, files: List[str]):
        self._files = files
        self._lines = []
        for i, path in enumerate(tqdm.tqdm(self._files)):
            file = open(path, mode="r", encoding="utf8")
            lines = file.readlines()
            segments = []
            text = ""
            for line in lines:
                line = line.rstrip('\n')
                if len(text) == 0:
                    text = line
                    continue
                if len(text) + len(line) >= 512:
                    if len(segments) < 2:
                        segments.append(text)
                    else:
                        self._lines.append(segments)
                        segments = []
                    text = ""
                else:
                    text = text + " " + line
            file.close()

    def __dict__(self):
        data = {"sentence1": [], "sentence2": []}
        for i in range(len(self)):
            data['sentence1'].append(self[i][1]['sentence1'])
            data['sentence2'].append(self[i][1]['sentence2'])
        return data

    def __dict_no_format__(self):
        data = []
        for i in range(len(self)):
            data.append(self[i][1])
        return data

    def __len__(self):
        return len(self._lines)

    def __getitem__(self, index) -> T_co:
        return 'sentences', {'sentence1': self._lines[index][0], 'sentence2': self._lines[index][1]}


class WebcorpusInMemoryCharformer(WebcorpusInMemory):
    def __init__(self, files: List[str], max_length=1024):
        super(WebcorpusInMemoryCharformer, self).__init__(files)
        self._files = files
        self._lines = []
        self.max_length = max_length
        for i, path in enumerate(tqdm.tqdm(self._files)):
            file = open(path, mode="r", encoding="utf8")
            lines = file.readlines()
            segments = []
            text = ""
            for line in lines:
                line = line.rstrip('\n')
                if len(bytearray(text, encoding="utf8")) == 0:
                    text = line
                    continue
                if len(bytearray(text, encoding="utf8")) + len(bytearray(line, encoding="utf8")) >= self.max_length//2:
                    if len(segments) < 2:
                        segments.append(text)
                    else:
                        self._lines.append(segments)
                        segments = []
                    text = ""
                else:
                    text = text + " " + line
            file.close()



