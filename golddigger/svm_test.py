
import sqlite3
from sklearn import svm
# from sklearn.preprocessing import StandardScaler


inputs_query = "select chain, offer, market, repeater from main.trainHistory"
tests_query = "select id, chain, offer, market from main.testHistory"
attach_query = "attach database './data/transaction_offer_reduced.db' as tor"
test_query = "select min(id) from tor.transaction_offer_reduced"

tests_join_query = "select testH.id, testH.chain, testH.offer, testH.market,\
    t.company, t.brand, t.productsize, t.purchasequantity, t.purchaseamount\
    from main.testHistory as testH inner join tor.transaction_offer_reduced\
    as t on testH.id=t.id"

inputs_join_query = "select tH.chain, tH.offer, tH.market, t.company, t.brand,\
    t.productsize, t.purchasequantity, t.purchaseamount, tH.repeater from\
    main.trainHistory as tH inner join tor.transaction_offer_reduced\
    as t on tH.id=t.id"


def clean_training_result(result, ind=3):
    res_list = list(result)
    if res_list[ind] == u't':
        res_list[ind] = 1
    else:
        res_list[ind] = 0

    return map(float, res_list[:ind]), res_list[-1]


def clean_testing_result(result):
    res_list = list(result)
    return res_list[0], map(float, res_list[1:])


def get_accuracy(t_out, r_out):
    total_sim = 0
    for t, r in zip(t_out, r_out):
        if t == r:
            total_sim += 1

    return float(total_sim) / len(t_out)


def get_testing_data():
    conn = sqlite3.connect("data/testHistory.db")
    bad_mouth = conn.cursor()

    # bad_mouth.execute(attach_query)

    ids = list()
    inputs = list()
    for row in bad_mouth.execute(tests_query):
        buyer_id, inp = clean_testing_result(row)
        ids.append(buyer_id)
        inputs.append(inp)

    return ids, inputs


def run_svm():
    conn = sqlite3.connect("data/trainHistory.db")
    profanity = conn.cursor()
    # profanity.execute(inputs_query)

    ids, test_inputs = get_testing_data()
    inputs = list()
    outputs = list()
    for i, row in enumerate(profanity.execute(inputs_query)):
        inp, out = clean_training_result(row)
        inputs.append(inp)
        outputs.append(out)

    clf = svm.SVC(max_iter=100)
    print "Fitting..."
    clf.fit(inputs, outputs)

    preds = clf.predict(test_inputs)
    write_kaggle_file(ids, preds)
    conn.commit()
    conn.close()


def write_kaggle_file(ids, preds):
    with open("output.csv", "w") as f:
        f.write("id,repeatProbability\n")
        for buyer_id, pred in zip(ids, preds):
            f.write("{},{}\n".format(buyer_id, pred))


if __name__ == "__main__":
    run_svm()
