import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets('./', one_hot=True)


# find non-zero fitness for selection
def get_fitness(pred): return pred


def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = np.random.normal(scale=0.1)
    return child


num_classes = 10
input_size = 784
hidden_units_size = 30
batch_size = 100
pop = np.array([[]])
training_iterations = 10000

W1_len = input_size * hidden_units_size
W2_len = hidden_units_size * num_classes

DNA_SIZE = W1_len + W2_len + num_classes + hidden_units_size          # DNA length
POP_SIZE = 50           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 200

for i in range(POP_SIZE):
    DNA = np.random.normal(size=[DNA_SIZE], scale=0.1)
    if i == 0:
        pop = np.append(pop[0], DNA, axis=0)
    else:
        pop = np.vstack((pop, DNA))

sess = tf.Session()

X = tf.placeholder(tf.float32, shape=[None, input_size])
Y = tf.placeholder(tf.float32, shape=[None, num_classes])
W1 = tf.Variable(tf.random_normal([input_size, hidden_units_size], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1, shape=[hidden_units_size]))
W2 = tf.Variable(tf.random_normal([hidden_units_size, num_classes], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, shape=[num_classes]))

hidden_opt = tf.matmul(X, W1) + B1
hidden_opt = tf.nn.relu(hidden_opt)
final_opt = tf.matmul(hidden_opt, W2) + B2
final_opt = tf.nn.relu(final_opt)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=final_opt))
opt = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(final_opt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess.run(init)

count = 0  # count training time

while True:
    fit_group = np.array([])
    # batch = mnist.train.next_batch(batch_size)
    for i in range(POP_SIZE):
        train_accuracy = [0, 0]

        W1 = pop[i, 0:W1_len].reshape((input_size, hidden_units_size))
        B1 = pop[i, W1_len:W1_len + hidden_units_size]
        W2 = pop[i, W1_len + hidden_units_size:W1_len + hidden_units_size + W2_len].reshape((hidden_units_size, num_classes))
        B2 = pop[i, -num_classes:]
        fit_avg = np.array([])
        for _ in range(100):
            batch = mnist.train.next_batch(batch_size)
            batch_input = batch[0]
            batch_labels = batch[1]
            train_accuracy = sess.run([opt, accuracy], feed_dict={
                X: batch_input, Y: batch_labels})
            fit_avg = np.hstack((fit_avg, train_accuracy[1]))
        pop[i, 0:W1_len] = sess.run(tf.reshape(W1, [1, -1]))
        pop[i, W1_len:W1_len + hidden_units_size] = sess.run(tf.reshape(B1, [1, -1]))
        pop[i, W1_len + hidden_units_size:W1_len + hidden_units_size + W2_len] = sess.run(tf.reshape(W2, [1, -1]))
        pop[i, -num_classes:] = sess.run(tf.reshape(B2, [1, -1]))
        fit_acc = fit_avg.sum()/100
        fit_group = np.hstack((fit_group, fit_acc))

    avg_fit = fit_group.sum()/POP_SIZE
    print("Average fitness: ", avg_fit)
    fitness = get_fitness(fit_group)
    max_fit = fit_group[np.argmax(fit_group)]
    print("Highest accurate rate: ", max_fit)
    plt.scatter(count*100, max_fit, s=200, lw=0, c='red', alpha=0.5)
    count += 1
    if max_fit > 0.98:
        break
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        i = 0
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child
        i += 1

plt.show()
print("BP begin:")

fittest_DNA = np.argmax(fitness)
W1 = pop[fittest_DNA, 0:W1_len].reshape((input_size, hidden_units_size))
B1 = pop[fittest_DNA, W1_len:W1_len + hidden_units_size]
W2 = pop[fittest_DNA, W1_len + hidden_units_size:W1_len + hidden_units_size + W2_len].reshape((hidden_units_size, num_classes))
B2 = pop[fittest_DNA, -num_classes:]
acc_avg = np.array([])
for i in range(training_iterations):
    batch = mnist.train.next_batch(batch_size)
    batch_input = batch[0]
    batch_labels = batch[1]
    train_accuracy = sess.run([opt, accuracy], feed_dict={X: batch_input, Y: batch_labels})
    acc_avg = np.hstack((acc_avg, train_accuracy[1]))
    if i % 1000 == 0:
        avg_acc = acc_avg.sum()/1000
        acc_avg = np.array([])
        print("Step : %d, training accuracy = %g " % (i, avg_acc))

acc_avg = np.array([])
avg_acc = 0
for _ in range(3000):
    batch = mnist.test.next_batch(batch_size)
    batch_input = batch[0]
    batch_labels = batch[1]
    train_accuracy = sess.run([opt, accuracy], feed_dict={X: batch_input, Y: batch_labels})
    acc_avg = np.hstack((acc_avg, train_accuracy[1]))
    avg_acc = acc_avg.sum() / 3000
print("Test accuracy %g" % avg_acc)

sess.close()
