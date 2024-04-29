from pathlib import Path
from typing import Optional, Sequence, Any, Union, Tuple

import torch
from torch.utils.data import ConcatDataset, Subset

from avalanche.benchmarks.classic.comniglot import _default_omniglot_train_transform, \
    _default_omniglot_eval_transform
from avalanche.benchmarks import nc_benchmark, EagerCLStream, CLScenario, DatasetExperience
from avalanche.benchmarks.utils import DataAttribute, AvalancheDataset, concat_datasets

from os.path import join
from typing import Optional, Callable

from torchvision.datasets import Omniglot

from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.utils.dataset_definitions import IDataset
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


def domain_incremental_omniglot_alphabets(
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ):
    """Targets are Omniglot alphabets. For each alphabet, we see one character at a time.
    
    Since the target is the alphabet, the resulting class is a mixture (of characters).
    """
    if transform is None:
        transform = Compose([Resize(28), ToTensor(), Normalize((0.9221,), (0.2681,))])
    train_data = AlphabetOmniglot(train=True, transform=transform, target_transform=target_transform, download=download)
    tgs = DataAttribute(train_data.targets, "targets")
    dms = DataAttribute(train_data.domains, "domains")
    train_data = AvalancheDataset([train_data], data_attributes=[tgs, dms])

    test_data = AlphabetOmniglot(train=False, transform=transform, target_transform=target_transform, download=download)
    tgs = DataAttribute(test_data.targets, "targets")
    dms = DataAttribute(test_data.domains, "domains")
    test_data = AvalancheDataset([test_data], data_attributes=[tgs, dms])

    max_doms = 0
    train_targets = np.array(train_data.targets)
    train_domains = np.array(train_data.domains)
    train_idxs = np.array(list(range(len(train_data))))
    for aid in range(max(train_data.targets)):
        doms_c = train_domains[train_targets == aid]
        nd = max(doms_c)
        max_doms = max(max_doms, nd)

    test_domains = np.array(test_data.domains)
    test_idxs = np.array(list(range(len(test_data))))

    eid = 0
    train_stream, test_stream = [], []
    # itera al contrario così le prime exp hanno meno domini, mentre le ultime hanno tutti gli alfabeti
    for idom in reversed(range(1, max_doms)):
        train_mask = (train_domains == idom)   
        train_dd = train_data.subset(train_idxs[train_mask])
        train_exp = DatasetExperience(dataset=train_dd, current_experience=eid)
        train_stream.append(train_exp)

        test_mask = (test_domains == idom)
        test_dd = test_data.subset(test_idxs[test_mask])
        test_exp = DatasetExperience(dataset=test_dd, current_experience=eid)
        test_stream.append(test_exp)

        eid += 1

    train_stream = EagerCLStream("train", train_stream)
    test_stream = EagerCLStream("test", test_stream)
    return CLScenario([train_stream, test_stream])


def SplitAlphabetOmniglot(
    n_experiences: int,
    *,
    return_task_id=False,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = True,
    train_transform: Optional[Any] = _default_omniglot_train_transform,
    eval_transform: Optional[Any] = _default_omniglot_eval_transform,
    dataset_root: Union[str, Path] = None
):
    """Class-incremental OMNIGLOT with the alphabet used as target.

    The scenario follows the one used by
        Rao, Dushyant, et al. "Continual unsupervised representation learning."
        Advances in Neural Information Processing Systems 32 (2019).

    which uses the alphabet as target instead of the character.

    The Omniglot dataset comprises 20 samples from each of 1623 characters,
    grouped into 50 different alphabets. For each character, we use 15 samples
    for the training set and 5 for the test set.
    *We use the 50 alphabets* as the class labels for evaluation.

    If needed, the dataset is automatically downloaded.

    :param n_experiences: The number of incremental experiences in the current
        benchmark. The value of this parameter should be a divisor of 10.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to True.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'omniglot' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """
    train_data = AlphabetOmniglot(train=True, download=True)
    test_data = AlphabetOmniglot(train=False, download=True)

    return nc_benchmark(
        train_dataset=train_data,
        test_dataset=test_data,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        class_ids_from_zero_in_each_exp=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


def get_alphabet_targets(data):
    aname_to_aid = {}  # alphabet name -> alphabet ID
    for idx, aname in enumerate(data._alphabets):
        aname_to_aid[aname] = idx

    cid_to_aid = {}  # class ID -> alphabet ID
    for idx, cname in enumerate(data._characters):
        aname = cname.split('/')[0]
        cid_to_aid[idx] = aname_to_aid[aname]

    alphabet_targets = []
    domain_targets = []
    for x in data._flat_character_images:
        cid = x[1]
        # convert class ID to alphabet ID
        alphabet_targets.append(cid_to_aid[cid])
        dom = int(data._characters[cid][-2:])
        domain_targets.append(dom)

    return alphabet_targets, domain_targets


class AlphabetOmniglot(IDataset):
    """Custom class used to adapt Omniglot (from Torchvision) and make it
    compatible with the Avalanche API.
    """

    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        dataset_root = default_dataset_location("omniglot")
        train_data = Omniglot(root=dataset_root, background=True, download=download)
        test_data = Omniglot(root=dataset_root, background=False, download=download)

        train_targets, train_domains = get_alphabet_targets(train_data)
        test_targets, test_domains = get_alphabet_targets(test_data)

        # test targets in Omniglot start from zero but we need different IDs
        # for train and test characters so we remap them.
        max_id = max(train_targets) + 1
        test_targets = [el + max_id for el in test_targets]

        # split train/test samples. for each character, 
        # we reserve 15 samples for train, 5 for test
        # we exploit the fact that samples are ordered by character
        targets = train_targets + test_targets
        domains = train_domains + test_domains
        data = ConcatDataset([train_data, test_data])

        mask = torch.zeros(20, dtype=torch.bool)
        mask[:15] = 1

        num_chars = len(train_data._characters) + len(test_data._characters)
        mask = mask.repeat(num_chars)
        idxs = torch.arange(0, len(data), dtype=torch.long)

        if self.train:
            self.data = Subset(data, idxs[mask])
            self.targets = torch.tensor(targets)[idxs[mask]].tolist()
            self.domains = torch.tensor(domains)[idxs[mask]].tolist()
        else:
            self.data = Subset(data, idxs[torch.logical_not(mask)])
            self.targets = torch.tensor(targets)[idxs[torch.logical_not(mask)]].tolist()
            self.domains = torch.tensor(domains)[idxs[torch.logical_not(mask)]].tolist()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target alphabet class.
        """
        x, _ = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y


if __name__ == "__main__":
    train_data = AlphabetOmniglot(train=True)
    test_data = AlphabetOmniglot(train=False)
    print(len(train_data), len(test_data))
    assert len(train_data) + len(test_data) == 32460

    bm = SplitAlphabetOmniglot(n_experiences=10)

    for e in bm.train_stream:
        print(len(e.dataset), e.classes_in_this_experience)
    for e in bm.test_stream:
        print(len(e.dataset), e.classes_in_this_experience)
