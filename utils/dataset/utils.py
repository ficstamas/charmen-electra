import os
import numpy as np
import re


def construct_paths(path: str, rgx: str):
    """
    Creates a paths of files
    :param path:
    :param rgx:
    :return:
    """
    regex_string = re.compile(rgx) if rgx is not None else None
    paths = []
    for _, _, f in os.walk(path):
        for x in f:
            if regex_string is None or regex_string.match(x) is not None:
                paths.append(os.path.join(path, x))
    return paths


def train_test_split(l: list, seed, p=0.8):
    state = np.random.get_state()
    np.random.seed(seed)

    train = np.random.choice(l, np.min([np.ceil(len(l)*p), len(l)]).astype(np.int64), replace=False)
    test = np.setdiff1d(l, train)

    np.random.set_state(state)
    return train.tolist(), test.tolist()
