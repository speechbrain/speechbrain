"""A library for encouraging models with multiple inputs to pay more
attention to inputs of interest

Authors
 * Artem Ploujnikov 2021
"""

import torch
from collections import namedtuple

ConditioningFeature = namedtuple('ConditioningFeature', 'name weight distortion')
ConditioningPredictions = namedtuple('ConditioingPredictions', 'actual distorted')
FeatureLoss = namedtuple(
    "FeatureLoss",
    "name raw_loss difference_loss weighted_difference_loss")

GarbageConditioningLoss = namedtuple(
    'ConditioningLoss',
    'effective_loss raw_loss garbage_loss feature_loss'
)


class GarbageConditioner:
    """
    A conditioning approach that attempts to penalize a model for
    learning a function that ignores key inputs because it is much
    "easier" to achieve good performance by focusing on the other
    inputs. One notable example is the training of an autoregressive
    Text-to-Speech model that is given both the input text sequences
    and the target spectrogram and learns to simply predict the
    the next step in a spectrogram based on previous steps, thus
    learning a generative model for speech instead of TTS.

    It is based on computing a surrogate loss as follows:

    For each included feature:
        Run the model with the input for that feature distored in the specified way
        Compute the difference between the "raw" loss (with all featurs intact) and the loss with the distortion
        Multiply the difference by the weight provided
    """

    def __init__(self, features, criterion, model):
        self.features = {feature.name: feature for feature in features}
        self.criterion = criterion
        self.model = model

    def compute_forward(self, batch):
        """
        Computes the forward pass

        Arguments
        ---------s
        batch: object
            a batch. The format of the batch is model-specific
            (PaddedBatch, a tensor, a tuple, etc)

        Returns
        -------
        predictions: ConditioningPredictions
        """
        #TODO: This is the naive implementation. It can be
        #optimized by collating the distorted batches into a
        #single batch
        return ConditioningPredictions(
            actual=self.model(batch),
            distorted={
                feature_name: self.get_feature_distorted_predictions(
                    feature_name, batch)
                for feature_name in self.features
            }
        )

    def get_feature_distorted_predictions(self, feature_name, batch):
        """
        Obtains predictions where the specified feature is
        distorted

        Arguments
        ---------
        feature_name: str
            the name of the conditioning feature to distort
        batch: object
            a batch

        Returns
        -------
        predictions: object
            the predictions
        """
        feature = self.features[feature_name]
        distorted_features = feature.distortion(batch)
        return self.model(distorted_features)


    def compute_objectives(self, predictions, targets):
        """
        Computes the "garbage-conditioned" surrogate loss.
        For most use cases, it is recommended to use
        compute_objectives_detailed to obtain and track
        a detailed breakdown of the losses

        Arguments
        ---------
        predictions: ConditioningPredictions
            Predictions based on the true input and distored
            inputs for each feature, as computed in compute_forward

        targets: object
            the true targets. The type may vary per model

        Returns
        -------
        loss: tensor
            the surrogate loss
        """
        objectives = self.compute_objectives_detailed(predictions, targets)
        return objectives.effective_loss

    def compute_objectives_detailed(self, predictions, targets):
        """
        Computes the "garbage-conditioned" surrogate loss and
        returns a GarbageConditioningLoss ejctiob
        """
        raw_loss = self.criterion(predictions.actual, targets)
        feature_losses = {
            feature_name: self.get_feature_loss(
                feature_name, predictions, targets, raw_loss)
            for feature_name in self.features}
        garbage_loss = torch.sum(
            torch.tensor(
                [feature.weighted_difference_loss
                for feature in feature_losses.values()]))
        return GarbageConditioningLoss(
            raw_loss=raw_loss,
            effective_loss=raw_loss + garbage_loss,
            garbage_loss=garbage_loss,
            feature_loss=feature_losses
        )

    def get_feature_loss(self, feature_name, predictions, targets, raw_loss):
        """
        Calculates the loss for a specific feature

        Arguments
        ---------
        feature_name: str
            The name of the feature
        predictions: object
            Model predictions. The type of object used is
            model-dependent (tensors, dicts, tuples or custom objects)
        targets: object
            the targets (ground truth data)
        raw_loss: tensor
            the raw loss without any distortion applied

        Returns
        -------
        loss: FeatrueLoss
            the detailed breakdown of the loss for the feature
        """
        feature = self.features[feature_name]
        loss_with_distortion = self.criterion(
            predictions.distorted[feature_name], targets)

        difference_loss = raw_loss - loss_with_distortion

        return FeatureLoss(
            name=feature.name,
            raw_loss=loss_with_distortion,
            difference_loss=difference_loss,
            weighted_difference_loss=feature.weight * difference_loss
        )
