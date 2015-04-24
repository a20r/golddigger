
# from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
from progressbar import ProgressBar
import table


class GroupLearner(object):

    def __init__(self, num_groups, data_table):
        self.num_groups = num_groups
        self.dt = data_table

    def learn(self, classifier, **kwargs):
        self.models = list()
        progress = ProgressBar()
        for i in progress(xrange(self.num_groups)):
            inputs = self.dt.get_training_inputs(i)
            outputs = self.dt.get_training_outputs(i)
            clf = classifier(**kwargs)
            clf.fit(inputs, outputs)
            self.models.append(clf)
        return self

    def predict(self, inputs):
        ret_sum = 0.0
        progress = ProgressBar()
        print "Predicting..."
        for i in progress(xrange(len(self.models))):
            ret_sum += self.models[i].predict(inputs)
        return ret_sum / len(self.models)

    def compute_auc(self):
        h_inputs = self.dt.get_unseen_inputs()
        h_outputs = self.dt.get_unseen_outputs()
        pred_out = self.predict(h_inputs)
        return roc_auc_score(h_outputs, pred_out)

    def generate_kaggle_file(self, kaggle_file):
        test_inputs = self.dt.get_test_inputs()
        ids = self.dt.get_test_ids()
        preds = self.predict(test_inputs)
        progress = ProgressBar()
        print "Writing Kaggle file..."
        with open(kaggle_file, "w") as f:
            f.write("id,repeatProbability\n")
            for i in progress(xrange(len(preds))):
                f.write("{},{}\n".format(ids[i], preds[i]))
        return self


if __name__ == "__main__":
    num_groups = 400
    kaggle_file = "output.csv"
    print "Loading data..."
    dt = table.DataTable().load(num_groups)
    print "Learning..."
    df = GroupLearner(num_groups, dt)
    df.learn(linear_model.RidgeClassifierCV,
             class_weight={1: 2, 0: 1})\
        .generate_kaggle_file(kaggle_file)
