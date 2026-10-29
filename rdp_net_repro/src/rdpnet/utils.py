import random
from typing import List

import numpy as np
import torch


def set_seed(seed: int = 3529) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label2name(label: List[int]) -> str:
    # Mapping copied from notebook cells
    name = 'C0'
    if label == [0, 0, 0, 0, 0, 0, 0, 0]: name = 'C1'
    elif label == [1, 0, 0, 0, 0, 0, 0, 0]: name = 'C2'
    elif label == [0, 1, 0, 0, 0, 0, 0, 0]: name = 'C3'
    elif label == [0, 0, 1, 0, 0, 0, 0, 0]: name = 'C4'
    elif label == [0, 0, 0, 1, 0, 0, 0, 0]: name = 'C5'
    elif label == [0, 0, 0, 0, 1, 0, 0, 0]: name = 'C6'
    elif label == [0, 0, 0, 0, 0, 1, 0, 0]: name = 'C7'
    elif label == [0, 0, 0, 0, 0, 0, 1, 0]: name = 'C8'
    elif label == [0, 0, 0, 0, 0, 0, 0, 1]: name = 'C9'
    elif label == [1, 0, 1, 0, 0, 0, 0, 0]: name = 'C10'
    elif label == [1, 0, 0, 1, 0, 0, 0, 0]: name = 'C11'
    elif label == [1, 0, 0, 0, 1, 0, 0, 0]: name = 'C12'
    elif label == [1, 0, 0, 0, 0, 0, 1, 0]: name = 'C13'
    elif label == [0, 1, 1, 0, 0, 0, 0, 0]: name = 'C14'
    elif label == [0, 1, 0, 1, 0, 0, 0, 0]: name = 'C15'
    elif label == [0, 1, 0, 0, 1, 0, 0, 0]: name = 'C16'
    elif label == [0, 1, 0, 0, 0, 0, 1, 0]: name = 'C17'
    elif label == [0, 0, 1, 0, 1, 0, 0, 0]: name = 'C18'
    elif label == [0, 0, 1, 0, 0, 0, 1, 0]: name = 'C19'
    elif label == [0, 0, 0, 1, 1, 0, 0, 0]: name = 'C20'
    elif label == [0, 0, 0, 1, 0, 0, 1, 0]: name = 'C21'
    elif label == [0, 0, 0, 0, 1, 0, 1, 0]: name = 'C22'
    elif label == [1, 0, 1, 0, 1, 0, 0, 0]: name = 'C23'
    elif label == [1, 0, 1, 0, 0, 0, 1, 0]: name = 'C24'
    elif label == [1, 0, 0, 1, 1, 0, 0, 0]: name = 'C25'
    elif label == [1, 0, 0, 1, 0, 0, 1, 0]: name = 'C26'
    elif label == [1, 0, 0, 0, 1, 0, 1, 0]: name = 'C27'
    elif label == [0, 1, 1, 0, 1, 0, 0, 0]: name = 'C28'
    elif label == [0, 1, 1, 0, 0, 0, 1, 0]: name = 'C29'
    elif label == [0, 1, 0, 1, 1, 0, 0, 0]: name = 'C30'
    elif label == [0, 1, 0, 1, 0, 0, 1, 0]: name = 'C31'
    elif label == [0, 1, 0, 0, 1, 0, 1, 0]: name = 'C32'
    elif label == [0, 0, 1, 0, 1, 0, 1, 0]: name = 'C33'
    elif label == [0, 0, 0, 1, 1, 0, 1, 0]: name = 'C34'
    elif label == [1, 0, 1, 0, 1, 0, 1, 0]: name = 'C35'
    elif label == [1, 0, 0, 1, 1, 0, 1, 0]: name = 'C36'
    elif label == [0, 1, 1, 0, 1, 0, 1, 0]: name = 'C37'
    elif label == [0, 1, 0, 1, 1, 0, 1, 0]: name = 'C38'
    return name

