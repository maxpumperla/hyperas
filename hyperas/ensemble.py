import numpy as np
from keras.models import model_from_yaml


class VotingModel(object):

    def __init__(self, model_list, voting='hard',
                 weights=None, nb_classes=None):
        """(Weighted) majority vote model for a given list of Keras models.

        Parameters
        ----------
        model_list: An iterable of Keras models.
        voting: Choose 'hard' for straight-up majority vote of highest model probilities or 'soft'
            for a weighted majority vote. In the latter, a weight vector has to be specified.
        weights: Weight vector (numpy array) used for soft majority vote.
        nb_classes: Number of classes being predicted.

        Returns
        -------
        A voting model that has a predict method with the same signature of a single keras model.
        """
        self.model_list = model_list
        self.voting = voting
        self.weights = weights
        self.nb_classes = nb_classes

        if voting not in ['hard', 'soft']:
            raise 'Voting has to be either hard or soft'

        if weights is not None:
            if len(weights) != len(model_list):
                raise ('Number of models {0} and length of weight vector {1} has to match.'
                       .format(len(weights), len(model_list)))

    def predict(self, X, batch_size=128, verbose=0):
        predictions = list(map(lambda model: model.predict(X, batch_size, verbose), self.model_list))
        nb_preds = len(X)

        if self.voting == 'hard':
            for i, pred in enumerate(predictions):
                pred = list(map(
                    lambda probas: np.argmax(probas, axis=-1), pred
                ))
                predictions[i] = np.asarray(pred).reshape(nb_preds, 1)
            argmax_list = list(np.concatenate(predictions, axis=1))
            votes = np.asarray(list(
                map(lambda arr: max(set(arr)), argmax_list)
            ))
        if self.voting == 'soft':
            for i, pred in enumerate(predictions):
                pred = list(map(lambda probas: probas * self.weights[i], pred))
                predictions[i] = np.asarray(pred).reshape(nb_preds, self.nb_classes, 1)
            weighted_preds = np.concatenate(predictions, axis=2)
            weighted_avg = np.mean(weighted_preds, axis=2)
            votes = np.argmax(weighted_avg, axis=1)

        return votes


def voting_model_from_yaml(yaml_list, voting='hard', weights=None):
    model_list = map(lambda yml: model_from_yaml(yml), yaml_list)
    return VotingModel(model_list, voting, weights)
