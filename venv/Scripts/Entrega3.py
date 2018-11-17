import tensorflow as tf
import numpy as np

csv = []

file = open(r'C:\Users\Cristian\PycharmProjects\AI-Project\input\Normalizado.csv', 'r')
for i in file.readlines():
    csv.append(i.split(';'))

# number of features
num_features = len(csv[0])
# number of target labels
num_labels = 4
# learning rate (alpha)
learning_rate = 0.05
# batch size
batch_size = 1
# number of epochs
num_steps = len(csv)

# input data
data = []
for line in csv:
    data.append([])
    for i in line:
        data[-1].append(tf.float32(i))
data = np.array(data)

INGC = np.array([1,0,0,0])
INGE = np.array([0,1,0,0])
INGO = np.array([0,0,1,0])
INGI = np.array([0,0,0,1])

labelSet = []
for i in range(77):
    labelSet.append(INGC)
for i in range(77):
    labelSet.append(INGE)
for i in range(77):
    labelSet.append(INGO)
for i in range(77):
    labelSet.append(INGI)

labelSet = np.array(labelSet)

#70% train 20% test 10%valid
idx70 = int((len(data)/100.0)*70)
idx90 = int((len(data)/100.0)*90)

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
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	# Variables.
	weights = tf.Variable(tf.truncated_normal([num_features, num_labels]))
	biases = tf.Variable(tf.zeros([num_labels]))

	# Training computation.
	logits = tf.matmul(tf_train_dataset, weights) + biases
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
	test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
