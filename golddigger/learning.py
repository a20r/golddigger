
# from sklearn import tree
from sklearn import ensemble
import table


class GroupLearner(object):

    def __init__(self, num_groups, data_table):
        self.num_groups = num_groups
        self.dt = data_table
        self.weights = list()

    def learn(self, classifier, **kwargs):
        self.dt.load(self.num_groups)
        self.models = list()
        errors = list()
        error_sum = 0.0
        h_inputs = self.dt.get_holdout_inputs()
        h_outputs = self.dt.get_holdout_outputs()
        for i in xrange(self.num_groups):
            inputs = self.dt.get_training_inputs(i)
            outputs = self.dt.get_training_outputs(i)
            clf = classifier(**kwargs)
            clf.fit(inputs, outputs)
            error = self.get_error(clf, h_inputs, h_outputs)
            error_sum += error
            errors.append(error)
            self.models.append(clf)
        self.weights = map(lambda s: (error_sum - s) / error_sum, errors)
        return self

    def get_error(self, model, inps, outs):
        error = 0.0
        preds = model.predict(inps)
        for pred, out in zip(preds, outs):
            error += abs(out - pred)
        return error

    def predict(self, inputs):
        ret_sum = 0.0
        for i, model in enumerate(self.models):
            ret_sum += self.weights[i] * model.predict_proba(inputs)[:, 1]
        return ret_sum

    def generate_kaggle_file(self, kaggle_file):
        test_inputs = self.dt.get_test_inputs()
        ids = self.dt.get_test_ids()
        preds = self.predict(test_inputs)
        with open(kaggle_file, "w") as f:
            f.write("id,repeatProbability\n")
            for buyer_id, pred in zip(ids, preds):
                f.write("{},{}\n".format(buyer_id, pred))


if __name__ == "__main__":
    dt = table.DataTable()
    num_groups = 10
    kaggle_file = "output.csv"
    df = GroupLearner(num_groups, dt)
    df.learn(ensemble.ExtraTreesClassifier,
             n_estimators=50).generate_kaggle_file(kaggle_file)
