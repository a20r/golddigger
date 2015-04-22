
from sklearn import tree
from sklearn.metrics import roc_auc_score
import table


class GroupLearner(object):

    def __init__(self, num_groups, data_table):
        self.num_groups = num_groups
        self.dt = data_table

    def learn(self, classifier, **kwargs):
        print "Loading data..."
        self.dt.load(self.num_groups)
        print "Learning..."
        self.models = list()
        self.weights = list()
        self.probs = list()
        h_inputs = self.dt.get_holdout_inputs()
        h_outputs = self.dt.get_holdout_outputs()
        for i in xrange(self.num_groups):
            inputs = self.dt.get_training_inputs(i)
            outputs = self.dt.get_training_outputs(i)
            clf = classifier(**kwargs)
            clf.fit(inputs, outputs)
            preds = clf.predict(h_inputs)
            self.weights.append(1 - roc_auc_score(h_outputs, preds))
            self.models.append(clf)
        self.probs = map(lambda w: w / sum(self.weights), self.weights)
        return self

    def predict(self, inputs):
        ret_sum = 0.0
        for i, model in enumerate(self.models):
            ret_sum += self.probs[i] * model.predict_proba(inputs)[:, 1]
        return ret_sum

    def compute_auc(self):
        h_inputs = self.dt.get_unseen_inputs()
        h_outputs = self.dt.get_unseen_outputs()
        pred_out = self.predict(h_inputs)
        return roc_auc_score(h_outputs, pred_out)

    def generate_kaggle_file(self, kaggle_file):
        test_inputs = self.dt.get_test_inputs()
        ids = self.dt.get_test_ids()
        preds = self.predict(test_inputs)
        with open(kaggle_file, "w") as f:
            f.write("id,repeatProbability\n")
            for buyer_id, pred in zip(ids, preds):
                f.write("{},{}\n".format(buyer_id, pred))
        return self


if __name__ == "__main__":
    dt = table.DataTable()
    num_groups = 100
    kaggle_file = "output.csv"
    df = GroupLearner(num_groups, dt)
    print df.learn(tree.DecisionTreeClassifier)\
        .generate_kaggle_file(kaggle_file).compute_auc()
