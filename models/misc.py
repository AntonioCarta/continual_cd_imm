from avalanche.models.base_model import BaseModel
import torch
import torchvision.models as models
from avalanche.models import FeatureExtractorBackbone
from torch import nn


class CombinedModule(BaseModel, torch.nn.Module):

    def __init__(self, feature_extractor: BaseModel, classifier, train_feat_ext=False):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.train_feat_ext = train_feat_ext

    def get_features(self, x):
        if not self.train_feat_ext:
            self.feature_extractor.eval()
        return self.feature_extractor.get_features(x)

    def forward(self, x, y=None, skip_feature_extractor=False):
        if skip_feature_extractor:
            feat = x
        else:
            # extract feature
            feat = self.get_features(x)

        # fed the feat to the classifier
        return self.classifier(feat, y)


class FlattenInput(BaseModel, torch.nn.Module):

    def get_features(self, x):
        BS = x.shape[0]
        return x.view(BS, -1)

    def forward(self, x):
        return self.get_features(x)


class PretrainedFeatExtractor(BaseModel, FeatureExtractorBackbone):

    def __init__(self, arch_name, output_layer_name, hub_url=None):
        # find arch_name in torchvision first
        if arch_name in models.__dict__:
            pretrained_model = models.__dict__[arch_name](pretrained=True)
        else:
            # try to get the models from torch hub
            if hub_url is not None:
                pretrained_model = torch.hub.load(hub_url, arch_name, pretrained=True)
            else:
                raise ValueError(f"I don't know where to find {arch_name} model.")
        print(pretrained_model)
        super(PretrainedFeatExtractor, self).__init__(model=pretrained_model, output_layer_name=output_layer_name)

    def get_features(self, x):
        a = super(PretrainedFeatExtractor, self).forward(x)
        if a.ndim > 2:
            a_shape = a.shape
            a = a.reshape(a_shape[0], a_shape[1], -1)
            a = torch.mean(a, dim=2)
        return a

    def forward(self, x):
        return self.get_features(x)

'''
class Autoencoder(th.nn.Module):

    def __init__(self, input_size, repr_size):
        super().__init__()
        middle_size = (input_size + repr_size) // 2
        self.encoder = th.nn.Sequential(th.nn.Linear(input_size, middle_size), th.nn.ReLU(),
                                        th.nn.Linear(middle_size, middle_size), th.nn.ReLU(),
                                        th.nn.Linear(middle_size, repr_size))

        self.decoder = th.nn.Sequential(th.nn.Linear(repr_size, middle_size), th.nn.ReLU(),
                                        th.nn.Linear(middle_size, middle_size), th.nn.ReLU(),
                                        th.nn.Linear(middle_size, input_size))

    def forward(self, x):
        return self.encoder(x.view(x.shape[0], -1))

    def get_reconstruction_loss(self, x):
        out = self.decoder(self.encoder(x.view(x.shape[0], -1)))
        return F.mse_loss(x.view(x.shape[0], -1), out)
'''


def finn_cnn(num_classes):
    """follows [[finnModelagnosticMetalearningFast2017]] and Vinyals 2016."""
    return nn.Sequential(
        nn.Conv2d(1, 64, 3, stride = 2, padding = 1),
        # nn.BatchNorm2d(64, momentum=0.001),
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 64, 3, stride = 2, padding = 1),
        # nn.BatchNorm2d(64, momentum=0.001),
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 64, 3, stride = 2, padding = 1),
        # nn.BatchNorm2d(64, momentum=0.001),
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 64, 3, stride = 2, padding = 0),
        # nn.BatchNorm2d(64, momentum=0.001),
        nn.ReLU(inplace=True),

        nn.Flatten(),
        nn.Linear(64, num_classes)
    )
