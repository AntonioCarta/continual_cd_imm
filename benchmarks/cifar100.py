from avalanche.benchmarks import SplitCIFAR100
from avalanche.benchmarks.classic.ccifar100 import _default_cifar100_train_transform , _default_cifar100_eval_transform
from avalanche.benchmarks.datasets import default_dataset_location
from torchvision.datasets import CIFAR100
from avalanche.benchmarks.utils import TransformGroups, as_classification_dataset, DataAttribute, AvalancheDataset, concat_datasets
from avalanche.benchmarks.scenarios import _split_dataset_by_attribute, benchmark_from_datasets, benchmark_with_validation_stream
import random


coarse_to_fine = {  # coarse -> fine label mapping
    0: [72, 4, 95, 30, 55],  # aquatic mammals beaver, dolphin, otter, seal, whale
    1: [73, 32, 67, 91, 1],  # fish	aquarium fish, flatfish, ray, shark, trout
    2: [92, 70, 82, 54, 62], # flowers	orchids, poppies, roses, sunflowers, tulips
    3: [16, 61, 9, 10, 28],  # food containers	bottles, bowls, cans, cups, plates
    4: [51, 0, 53, 57, 83],  # fruit and vegetables	apples, mushrooms, oranges, pears, sweet peppers
    5: [40, 39, 22, 87, 86], # household electrical devices	clock, computer keyboard, lamp, telephone, television
    6: [20, 25, 94, 84, 5],  # household furniture	bed, chair, couch, table, wardrobe
    7: [14, 24, 6, 7, 18],   # insects	bee, beetle, butterfly, caterpillar, cockroach
    8: [43, 97, 42, 3, 88],  # large carnivores	bear, leopard, lion, tiger, wolf
    9: [37, 17, 76, 12, 68], # large man-made outdoor things	bridge, castle, house, road, skyscraper
    10: [49, 33, 71, 23, 60],# large natural outdoor scenes	cloud, forest, mountain, plain, sea
    11: [15, 21, 19, 31, 38],# large omnivores and herbivores	camel, cattle, chimpanzee, elephant, kangaroo
    12: [75, 63, 66, 64, 34],# medium-sized mammals	fox, porcupine, possum, raccoon, skunk
    13: [77, 26, 45, 99, 79],# non-insect invertebrates	crab, lobster, snail, spider, worm
    14: [11, 2, 35, 46, 98], # people	baby, boy, girl, man, woman
    15: [29, 93, 27, 78, 44],# reptiles	crocodile, dinosaur, lizard, snake, turtle
    16: [65, 50, 74, 36, 80],# small mammals	hamster, mouse, rabbit, shrew, squirrel
    17: [56, 52, 47, 59, 96],# trees	maple, oak, palm, pine, willow
    18: [8, 58, 90, 13, 48], # vehicles 1	bicycle, bus, motorcycle, pickup truck, train
    19: [81, 69, 41, 89, 85] # vehicles 2	lawn-mower, rocket, streetcar, tank, tractor
}

fine_to_coarse = {  # fine -> coarse label mapping
    72: 0, 4: 0, 95: 0, 30: 0, 55: 0, 
    73: 1, 32: 1, 67: 1, 91: 1, 1: 1, 
    92: 2, 70: 2, 82: 2, 54: 2, 62: 2, 
    16: 3, 61: 3, 9: 3, 10: 3, 28: 3, 
    51: 4, 0: 4, 53: 4, 57: 4, 83: 4, 
    40: 5, 39: 5, 22: 5, 87: 5, 86: 5, 
    20: 6, 25: 6, 94: 6, 84: 6, 5: 6, 
    14: 7, 24: 7, 6: 7, 7: 7, 18: 7, 
    43: 8, 97: 8, 42: 8, 3: 8, 88: 8, 
    37: 9, 17: 9, 76: 9, 12: 9, 68: 9, 
    49: 10, 33: 10, 71: 10, 23: 10, 60: 10, 
    15: 11, 21: 11, 19: 11, 31: 11, 38: 11, 
    75: 12, 63: 12, 66: 12, 64: 12, 34: 12, 
    77: 13, 26: 13, 45: 13, 99: 13, 79: 13, 
    11: 14, 2: 14, 35: 14, 46: 14, 98: 14, 
    29: 15, 93: 15, 27: 15, 78: 15, 44: 15, 
    65: 16, 50: 16, 74: 16, 36: 16, 80: 16, 
    56: 17, 52: 17, 47: 17, 59: 17, 96: 17, 
    8: 18, 58: 18, 90: 18, 13: 18, 48: 18, 
    81: 19, 69: 19, 41: 19, 89: 19, 85: 19
}

def cifar100_superclasses(num_experiences):
    def _label_transform(y):
        return fine_to_coarse[y]

    tgroup = TransformGroups({
        "train": _default_cifar100_train_transform,
        "eval": _default_cifar100_eval_transform
    })

    def _make_dset(data):
        """add coarse labels and set avalanche attributes and transform groups."""
        # create coarse labels
        fine_targets = data.targets
        coarse_targets = [fine_to_coarse[el] for el in fine_targets]

        # save coarse and fine labels as separate attributes in AvalancheDataset
        da1 = DataAttribute(fine_targets, "domains")
        da2 = DataAttribute(coarse_targets, "coarse_targets")
        da3 = DataAttribute(coarse_targets, "targets")

        data = AvalancheDataset(data, data_attributes=[da1, da2, da3])
        data = as_classification_dataset(data, tgroup)
        return data

    # load CIFAR100 dataset
    dataset_root = default_dataset_location("cifar100")
    
    train_set = _make_dset(CIFAR100(str(dataset_root), target_transform=_label_transform, train=True, download=True))
    test_set = _make_dset(CIFAR100(str(dataset_root), target_transform=_label_transform, train=False, download=True))

    # split stream by fine labels, showing exactly one fine class for each coarse class per experience
    train_sets = _split_dataset_by_attribute(train_set, "domains")
    test_sets = _split_dataset_by_attribute(test_set, "domains")

    fine_sets = list(set(train_sets.keys()))
    random.shuffle(fine_sets)
    assert len(fine_sets) == 100
    assert 100 % num_experiences == 0, "num_experiences must be divisible by 100."
    cls_per_exp = 100 // num_experiences

    train_stream, test_stream = [], []
    idx = 0
    for eid in range(num_experiences):
        curr_labels = [fine_sets[idx + k] for k in range(cls_per_exp)]

        td = concat_datasets([train_sets[y] for y in curr_labels])
        train_stream.append(td)
        td = concat_datasets([test_sets[y] for y in curr_labels])
        test_stream.append(td)

        idx += cls_per_exp

    bm = benchmark_from_datasets(
        train=train_stream,
        test=test_stream
    )
    return bm


if __name__ == "__main__":
    bm = cifar100_superclasses(10)

    for exp in bm.train_stream: 
        untgs = exp.dataset.targets.uniques
        undoms = exp.dataset.fine_targets.uniques
        # print(f"{exp.current_experience} - num.classes {len(untgs)} - domains {len(undoms)} - samples {len(exp.dataset)} - doms {undoms}")
        print(f"{exp.current_experience} - num.classes {len(untgs)} - samples {len(exp.dataset)}")
        print(f"\ttargets: {untgs}")
        print(f"\tdomains: {undoms}")
