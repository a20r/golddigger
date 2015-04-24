
# from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn import linear_model
from progressbar import ProgressBar
import tableFork


class GroupLearner(object):

    def __init__(self, num_groups, data_table):
        self.num_groups = num_groups
        self.dt = data_table

    def learn(self, classifier, **kwargs):
        self.models = list()
        self.weights = list()
        self.probs = list()
        self.pca = PCA().fit(self.dt.get_all_training_inputs())
        h_inputs = self.dt.get_holdout_inputs()
        h_outputs = self.dt.get_holdout_outputs()
        progress = ProgressBar()
        for i in progress(xrange(self.num_groups)):
            inputs = self.dt.get_training_inputs(i)
            outputs = self.dt.get_training_outputs(i)
            clf = classifier(**kwargs)
            clf.fit(inputs, outputs)
            preds = clf.predict(h_inputs)
            self.weights.append(roc_auc_score(h_outputs, preds))
            self.models.append(clf)
        self.probs = map(lambda w: w / sum(self.weights), self.weights)
        return self

    def predict(self, inputs):
        print "Predicting..."
        progress = ProgressBar()
        inputs = inputs
        ret_sum = 0.0
        for i, model in progress(enumerate(self.models)):
            ret_sum += self.probs[i] * model.predict_proba(inputs)[:, 1]
        return ret_sum

    def compute_auc(self):
        h_inputs = self.dt.get_unseen_inputs()
        h_outputs = self.dt.get_unseen_outputs()

        pred_out = self.predict(h_inputs)

        fpr, tpr, thresholds = roc_curve(h_outputs, pred_out)

        print fpr
        print tpr
        print thresholds
        
        return roc_auc_score(h_outputs, pred_out)

    def generate_kaggle_file(self, kaggle_file):
        compute_auc()

        print "Generating Kaggle file..."
        progress = ProgressBar()
        test_inputs = self.dt.get_test_inputs()
        ids = self.dt.get_test_ids()
        preds = self.predict(test_inputs)
        with open(kaggle_file, "w") as f:
            f.write("id,repeatProbability\n")
            for buyer_id, pred in progress(zip(ids, preds)):
                f.write("{},{}\n".format(buyer_id, pred))
        return self


class GroupLearnerClassifier(GroupLearner):
    def predict(self, inputs):
        inputs = inputs
        ret_sum = 0.0
        for i, model in enumerate(self.models):
            ret_sum += model.predict(inputs)
        return ret_sum / len(self.models)


if __name__ == "__main__":
    num_groups = 500
    kaggle_file = "output_com_final.csv"
    print "Loading data..."
    dt = tableFork.DataTable().load(num_groups)
    print "Learning..."
    df = GroupLearnerClassifier(num_groups, dt)
    df.learn(linear_model.RidgeClassifierCV,
             class_weight={1: 2, 0: 1})\
        .generate_kaggle_file(kaggle_file)
