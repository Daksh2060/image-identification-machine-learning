# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 


# Mira implementation
import util
PRINT = True


class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            cGrid = [0.001, 0.002, 0.004, 0.008]
        else:
            cGrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, cGrid)

    def scale_feature_vectors(self, data):

        scaled_data = []

        for datum in data:
            norm = sum(value ** 2 for value in datum.values()) ** 0.5
            scaled_datum = util.Counter({key: value / norm for key, value in datum.items()})
            scaled_data.append(scaled_datum)

        return scaled_data

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):

        scaled_training_data = self.scale_feature_vectors(trainingData)

        for C in Cgrid:

            for i in range(self.max_iterations):

                for j in range(len(scaled_training_data)):
                    score = util.Counter()

                    for label in self.legalLabels:
                        score[label] = self.weights[label] * scaled_training_data[j]

                    best_label = score.argMax()
                    correct_label = trainingLabels[j]

                    if best_label != correct_label:
                        f_square = sum(value ** 2 for value in scaled_training_data[j].values())
                        temp_t = ((self.weights[best_label] - self.weights[correct_label]) * scaled_training_data[
                            j] + 1.0) / (2 * f_square)
                        t = min(C, temp_t)

                        if not self.automaticTuning:
                            t = t / (1 + i)

                        for feature, value in scaled_training_data[j].items():
                            self.weights[best_label][feature] -= value * t
                            self.weights[correct_label][feature] += value * t

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        best = None
        for datum in data:
            vectors = util.Counter()
            for label in self.legalLabels:
                vectors[label] = self.weights[label] * datum
                best = vectors.argMax()
            guesses.append(best)
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """

        "*** YOUR CODE HERE ***"
        weights = self.weights[label]
        features_weights = weights.sortedKeys()[:100]

        return features_weights
