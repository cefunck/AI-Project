import tensorflow as tf
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime.now()
normalizado = []
labels = []
testDefinitivo = [] #aqui se cargaran los alumnos a predecir con la red entrenada
testDefinitivoLabels = [] #aqui se cargaran los alumnos a predecir con la red entrenada

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
learning_rate = 0.000005 #0.5
# Num nodes per hidden layer
num_nodes = [4000,400,4]
numHiddenLayers = len(num_nodes)
# batch size
batch_size = 100 #213 #150 #275
# number of epochs
num_steps = 8000 #10000 #2573
# data to plot
nameFig = r"\ learning_rate_"+str(learning_rate)+" batch_size_"+str(batch_size)+" num_steps_"+str(num_steps)+" numHiddenLayers_"+str(numHiddenLayers)+" numNodes_"+",".join([str(i) for i in num_nodes])
x = []
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

labelSet = np.array(labelSet)
data = np.array(data)
# 90% train 5% test 5%valid
idx90 = int((len(data) / 100.0) * 90)
idx95 = int((len(data) / 100.0) * 95)

train_dataset = data[:idx90]
train_labels = labelSet[:idx90]
valid_dataset = data[idx90:idx95]
valid_labels = labelSet[idx90:idx95]
test_dataset = data[idx95:]
test_labels = labelSet[idx95:]

# initialize a tensorflow graph
graph = tf.Graph()
W = []
B = []
logitsTrains = []
logitsValids = []
logitsTests = []

with graph.as_default():
    """ 
    defining all the nodes 
    """
    # Inputs
    logitsTrains.append(tf.placeholder(tf.float32, shape=(batch_size, num_features)))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    logitsValids.append(tf.constant(valid_dataset, tf.float32))
    logitsTests.append(tf.constant(test_dataset, tf.float32))

for i in range(numHiddenLayers):
    with graph.as_default():
        # Variables.
        if i == 0:
            num_inputs = 2079
        else:
            num_inputs = num_nodes[i-1]
        W.append(tf.Variable(tf.truncated_normal([num_inputs, num_nodes[i]])))
        B.append(tf.Variable(tf.zeros([num_nodes[i]])))

        # Training computation.
        logitsTrains.append(tf.matmul(logitsTrains[-1], W[-1]) + B[-1])
        logitsValids.append(tf.matmul(logitsValids[-1], W[-1]) + B[-1])
        logitsTests.append(tf.matmul(logitsTests[-1], W[-1]) + B[-1])

with graph.as_default():
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logitsTrains[-1])
    valid_prediction = tf.nn.softmax(logitsValids[-1])
    test_prediction = tf.nn.softmax(logitsTests[-1])
    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logitsTrains[-1]))
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


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
        feed_dict = {logitsTrains[0]: batch_data,tf_train_labels: batch_labels}

        # run one step of computation
        _,l,predictions = session.run([optimizer, loss, train_prediction],feed_dict=feed_dict)

        if (step % 100 == 0):
            print("Minibatch loss at step {0}: {1}".format(step, l))
            x.append(step)
            y = accuracy(predictions, batch_labels)
            print("Minibatch accuracy: {:.1f}%".format(y))
            y1.append(y)
            y = accuracy(valid_prediction.eval(), valid_labels)
            #plt.scatter(step, y2)
            print("Validation accuracy: {:.1f}%".format(y))
            y2.append(y)
            #if (y >= 85):  # 77.4 #80
                #break
            #plt.scatter(step, y2)
            #plt.pause(0.000000000000003)
    print("\nTest accuracy: {:.1f}%".format(accuracy(test_prediction.eval(), test_labels)))

end = datetime.datetime.now()
print("time",str(end-start))
plt.figure()
print(len(x),len(y1),len(y2))
plt.plot(x, y1, x, y2)
plt.legend(['Minibatch accuracy', 'Validation accuracy'])
nameFig = r"C:\Users\Cristian\PycharmProjects\AI-Project\registro plots"+nameFig+".png"
plt.savefig(nameFig)
plt.show()


