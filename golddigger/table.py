
import sqlite3


class DataTable(object):

    def __init__(self):
        self.train_query = "select * from main.features_train"
        self.test_query = "select * from main.features_test"
        self.db = "data/data.db"

    def load(self, num_groups=1):
        self.inputs = list()
        self.outputs = list()
        self.test_inputs = list()
        self.test_ids = list()
        self.group_inputs = [list() for _ in xrange(num_groups)]
        self.group_outputs = [list() for _ in xrange(num_groups)]

        with sqlite3.connect(self.db) as conn:
            cursor = conn.cursor()
            features_train = cursor.execute(self.train_query)
            for i, feature_vector in enumerate(features_train):
                f_vector = list(feature_vector)
                repeater = f_vector[1]
                del f_vector[1]
                del f_vector[0]
                f_vector = map(self.float_or_none, f_vector)
                self.inputs.append(f_vector)
                self.outputs.append(repeater)
                self.group_inputs[i % num_groups].append(f_vector)
                self.group_outputs[i % num_groups].append(repeater)

            features_test = cursor.execute(self.test_query)
            for feature_vector in features_test:
                f_vector = list(feature_vector)
                f_vector = map(self.float_or_none, f_vector)
                self.test_ids.append(f_vector[0])
                self.test_inputs.append(f_vector[1:])

        return self

    def float_or_none(self, v):
        return float(v) if v else 0.0

    def get_all_training_inputs(self):
        return self.inputs

    def get_all_training_outputs(self):
        return self.outputs

    def get_training_inputs(self, group):
        return self.group_inputs[group]

    def get_training_outputs(self, group):
        return self.group_outputs[group]

    def get_test_inputs(self):
        return self.test_inputs

    def get_test_ids(self):
        return self.test_ids
