import tensorflow as tf
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime.now()
normalizado = []
labels = []

file = open(r'NormalizadoOficial.csv', 'r')
for i in file.readlines():
    normalizado.append(i.split(';'))

file = open(r'labelsOficial.csv', 'r')
for i in file.readlines():
    labels.append(i[1:-2].split(" "))

# number of features
num_features = len(normalizado[0])
# number of target labels
num_labels = 4
# learning rate (alpha)
learning_rate = 0.1 #0.5
# batch size
batch_size = 50 #210
# number of epochs
num_steps = 10000 #10000

# data to plot
x = range(num_steps)
y1 = []
y2 = []


# input data
data = []
labelSet = []
for line in normalizado:
    data.append([])
    for i in line:
        data[-1].append(float(i))
for i in labels:
    onehot = []
    for j in i:
        onehot.append(float(j))
    onehot = np.array(onehot)
    labelSet.append(onehot)

INGC = np.array([1, 0, 0, 0])
INGE = np.array([0, 1, 0, 0])
INGO = np.array([0, 0, 1, 0])
INGI = np.array([0, 0, 0, 1])

labelSet = np.array(labelSet)
data = np.array(data)
# 70% train 20% test 10%valid
idx70 = int((len(data) / 100.0) * 70)
idx90 = int((len(data) / 100.0) * 90)

train_dataset = data[:idx70]
train_labels = labelSet[:idx70]
test_dataset = data[idx70:idx90]
test_labels = labelSet[idx70:idx90]
valid_dataset = data[idx90:]
valid_labels = labelSet[idx90:]

# initialize a tensorflow graph
graph = tf.Graph()

with graph.as_default():
    """ 
	defining all the nodes 
	"""

    # Inputs
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset, tf.float32)
    tf_test_dataset = tf.constant(test_dataset, tf.float32)

    # Variables.
    weights = tf.Variable(tf.truncated_normal([num_features, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


# utility function to calculate accuracy
def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu


with tf.Session(graph=graph) as session:
    # initialize weights and biases
    tf.global_variables_initializer().run()
    print("Initialized")

    for step in range(num_steps):
        # pick a randomized offset
        offset = np.random.randint(0, train_labels.shape[0] - batch_size - 1)

        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        # Prepare the feed dict
        feed_dict = {tf_train_dataset: batch_data,tf_train_labels: batch_labels}

        # run one step of computation
        _, l, predictions = session.run([optimizer, loss, train_prediction],feed_dict=feed_dict)

        if (step % 1 == 0):
            print("Minibatch loss at step {0}: {1}".format(step, l))
            y = accuracy(predictions, batch_labels)
            print("Minibatch accuracy: {:.1f}%".format(y))
            y1.append(y)
            y = accuracy(valid_prediction.eval(), valid_labels)
            print("Validation accuracy: {:.1f}%".format(y))
            y2.append(y)

    print("\nTest accuracy: {:.1f}%".format(accuracy(test_prediction.eval(), test_labels)))
end = datetime.datetime.now()

print("time",str(end-start))
plt.plot(x, y1, x, y2)
plt.legend(['Minibatch accuracy', 'Validation accuracy'])
plt.show()

