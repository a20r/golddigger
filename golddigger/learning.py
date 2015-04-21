
from sklearn import tree
import table


def decision_forests(num_groups):
    dt = table.DataTable()
    dt.load(num_groups)
    models = list()
    for i in xrange(num_groups):
        inputs = dt.get_training_inputs(i)
        outputs = dt.get_training_outputs(i)
        clf = tree.DecisionTreeClassifier()
        clf.fit(inputs, outputs)
        models.append(clf)

    test_inputs = dt.get_test_inputs()
    ids = dt.get_test_ids()
    preds = predict_df_avg(models, test_inputs)
    write_kaggle_file(ids, preds[:, 1])


def predict_df_avg(models, inputs):
    ret_sum = 0.0
    for model in models:
        ret_sum += model.predict_proba(inputs)
    return ret_sum / len(models)


def write_kaggle_file(ids, preds):
    with open("output.csv", "w") as f:
        f.write("id,repeatProbability\n")
        for buyer_id, pred in zip(ids, preds):
            f.write("{},{}\n".format(buyer_id, pred))


if __name__ == "__main__":
    decision_forests(5)
