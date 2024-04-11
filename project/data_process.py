import load_data as load_data

BATCH_SIZE = 84000
# return DataSet class
data = load_data.read_data_sets(one_hot=True)




# get train data and labels by batch size
train_x, train_label = data.train.next_batch(BATCH_SIZE) #change to just data.train?

# get test data
test_x = data.test.data

# get test labels
test_labels = data.test.labels

# get sample number
n_samples = data.train.num_examples


train_data = [(text, label) for text, label in zip(train_x, train_label)]
test_data = [(text, label) for text, label in zip(test_x, test_labels)]
