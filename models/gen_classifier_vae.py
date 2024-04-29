"""
Adaptation of Gido's Class VAE as in 
https://github.com/GMvandeVen/class-incremental-learning
"""
import copy
import numpy as np
import torch
from torch import nn
import numpy as np
import torch
from torch import nn


class GenClassifier(nn.Module):
    """Class for generative classifier with separate VAE for each class to be learned."""

    def __init__(self, vae_model, classes):

        # Set configurations for setting up the model
        super().__init__()
        self.classes = classes
        self.label = "GenClassifier"

        # Define a VAE for each class to be learned
        for class_id in range(classes):
            new_vae = copy.deepcopy(vae_model)
            setattr(self, "vae{}".format(class_id), new_vae)


    ##------ NAMES --------##

    def get_name(self):
        return "x{}-{}".format(self.classes, self.vae0.get_name())

    @property
    def name(self):
        return self.get_name()


    ##------ UTILITIES --------##

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


    ##------ SAMPLE FUNCTIONS --------##

    def sample(self, size, only_x=True, class_id=None, **kwargs):
        '''Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device.'''

        for sample_id in range(size):
            # sample from which class-specific VAE to sample
            selected_class_id = np.random.randint(0, self.classes, 1)[0] if class_id is None else class_id
            model_to_sample_from = getattr(self, 'vae{}'.format(selected_class_id))

            # sample from that VAE
            new_sample = model_to_sample_from.sample(1)

            # concatanate generated X (and y)
            X = torch.cat([X, new_sample], dim=0) if sample_id>0 else new_sample
            if not only_x:
                y = torch.cat([y, torch.LongTensor([selected_class_id]).to(self._device())]) if (
                    sample_id>0
                ) else torch.LongTensor([selected_class_id]).to(self._device())

        # return samples as [size]x[channels]x[image_size]x[image_size] tensor (and labels as [size] tensor)
        return X if only_x else (X, y)


    ##------ CLASSIFICATION FUNCTIONS --------##

    def classify(self, x, S=10, batch_size=128, allowed_classes=None):
        '''Given an input [x] (=batch-size must be 1!), return predicted class-ID (as integer).

        Input:  - [x]          <4D-tensor> of shape [1]x[channels]x[image_size]x[image_size]

        Output: - [predicated] <int> predicted class-ID
        '''

        # If not provided, set [allowed_classes] to all possible classes
        if allowed_classes is None:
            allowed_classes = list(range(self.classes))

        # For each possible class, evaluate likelihood of [x] under corresponding class-specific generative model
        scores = []
        with torch.no_grad():
            for class_id in allowed_classes:
                scores.append(
                    getattr(self, 'vae{}'.format(class_id)).estimate_loglikelihood_single(x, S=S, batch_size=batch_size)
                )

        # Extract [class_id] with highest likelihod and return
        scores = [el.item() for el in scores]
        predicted = allowed_classes[np.argmax(scores)]
        return predicted


