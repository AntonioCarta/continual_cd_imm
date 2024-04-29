from torchvision import transforms
from avalanche.benchmarks.classic import CORe50


def SLDA_CORe50(scenario='nc', run=0):
    # --- TRANSFORMATIONS
    _mu = [0.485, 0.456, 0.406]  # imagenet normalization
    _std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mu, std=_std),
        ]
    )
    # ---------

    # --- BENCHMARK CREATION
    return CORe50(
        scenario=scenario,
        train_transform=transform,
        eval_transform=transform,
        run=run
    )
