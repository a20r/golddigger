
import sqlite3
from sklearn import tree

inputs_created_query = "select * from main.features_train"
test_created_query = "select * from main.features_test"

inputs_query = "select chain, market, repeater from main.trainHistory"
inputs_join_offers_query = "select t.offer, t.market, o.quantity,\
    o.offervalue, t.repeater from main.trainHistory as\
    t inner join off.offers as o on t.offer=o.offer"
tests_join_offers_query = "select t.id, t.offer, t.market,\
    o.quantity, o.offervalue from main.testHistory as t inner\
    join off.offers as o on t.offer=o.offer"
tests_query = "select id, chain, offer, market from main.testHistory"
attach_query = "attach database './data/transaction_offer_reduced.db' as tor"
attach_offer = "attach database './data/offers.db' as off"
test_query = "select min(id) from tor.transaction_offer_reduced"

tests_join_query = "select testH.id, testH.chain, testH.offer, testH.market,\
    t.company, t.brand, t.productsize, t.purchasequantity, t.purchaseamount\
    from main.testHistory as testH inner join tor.transaction_offer_reduced\
    as t on testH.id=t.id"

inputs_join_query = "select tH.chain, tH.offer, tH.market, t.company, t.brand,\
    t.productsize, t.purchasequantity, t.purchaseamount, tH.repeater from\
    main.trainHistory as tH inner join tor.transaction_offer_reduced\
    as t on tH.id=t.id"

connect = "data/data.db"
old_connect = "data/trainHistory.db"


def clean_training_result(result, ind=51):

    res_list = list(result)

    temp = res_list[1]
    del res_list[1]
    res_list.append(temp)

    if res_list[ind] == u't':
        res_list[ind] = 1.0
    else:
        res_list[ind] = 0.0

    return map(lambda v: float(v) if v else 0.0, res_list[:ind]), res_list[-1]


def clean_testing_result(result):
    res_list = list(result)
    return res_list[0], map(lambda v: float(v) if v else 0.0, res_list[1:])


def get_testing_data():
    conn = sqlite3.connect(connect)
    bad_mouth = conn.cursor()
    # bad_mouth.execute(attach_offer)

    ids = list()
    inputs = list()
    for row in bad_mouth.execute(test_created_query):
        buyer_id, inp = clean_testing_result(row)
        ids.append(buyer_id)
        inputs.append(inp)

    return ids, inputs


def decision_forests(num_groups):
    conn = sqlite3.connect(connect)
    profanity = conn.cursor()
    # profanity.execute(attach_offer)

    ids, test_inputs = get_testing_data()
    inputs_list = [list() for _ in xrange(num_groups)]
    outputs_list = [list() for _ in xrange(num_groups)]
    for i, row in enumerate(profanity.execute(inputs_created_query)):
        inp, out = clean_training_result(row)
        inputs_list[i % num_groups].append(inp)
        outputs_list[i % num_groups].append(out)

    models = list()
    for inputs, outputs in zip(inputs_list, outputs_list):
        clf = tree.DecisionTreeClassifier()
        clf.fit(inputs, outputs)
        models.append(clf)

    preds = predict_df_avg(models, test_inputs)
    write_kaggle_file(ids, preds[:, 1])
    conn.commit()
    conn.close()


def predict_df_avg(models, inputs):
    ret_sum = 0.0
    for model in models:
        ret_sum += model.predict_proba(inputs)
    return ret_sum / len(models)


def predict_df_max(models, inputs):
    max_probs = models[0].predict_proba(inputs)
    for model in models:
        preds = model.predict_proba(inputs)
        for i, pred in enumerate(preds):
            if pred[1] > max_probs[i][1]:
                max_probs[i] = pred
    return max_probs


def write_kaggle_file(ids, preds):
    with open("output.csv", "w") as f:
        f.write("id,repeatProbability\n")
        for buyer_id, pred in zip(ids, preds):
            f.write("{},{}\n".format(buyer_id, pred))


if __name__ == "__main__":
    decision_forests(5)
