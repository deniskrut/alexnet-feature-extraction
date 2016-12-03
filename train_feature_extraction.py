import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from alexnet import AlexNet
import math

# TODO: Load traffic signs data.
nb_classes = 43

training_file = 'train.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']

encoder = LabelBinarizer()
encoder.fit(range(nb_classes))
y_train_bin = encoder.transform(y_train)

# TODO: Split data into training and validation sets.

train_features, valid_features, train_labels, valid_labels = train_test_split(X_train,
                                                                              y_train_bin,
                                                                              test_size=0.1,
                                                                              train_size=0.9,
                                                                              random_state=4242424)

# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(features, (227, 227))

labels = tf.placeholder(tf.float32, shape=[None, nb_classes])

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
biases = tf.Variable(tf.constant(0.05, shape=[nb_classes]))
fc_layer = tf.matmul(fc7, weights) + biases
prediction = tf.nn.softmax(fc_layer)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer,
                                                        labels=labels)
loss = tf.reduce_mean(cross_entropy)

prediction_argmax = tf.argmax(prediction, 1)
labels_argmax = tf.argmax(labels, 1)

is_correct_prediction = tf.equal(prediction_argmax, labels_argmax)

accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}

learning_rate = 0.0001
epochs = 20
batch_size = 256

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init = tf.initialize_all_variables()

session = tf.Session()

session.run(init)

batch_count = int(math.ceil(len(train_features)/batch_size))

for epoch_i in range(epochs):
    # The training cycle
    for batch_i in range(batch_count):
        # Get a batch of training features and labels
        batch_start = batch_i*batch_size
        batch_features = train_features[batch_start:batch_start + batch_size]
        batch_labels = train_labels[batch_start:batch_start + batch_size]
        
        # Run optimizer and get loss
        _, loss_res, accuracy_res = session.run(
                    [optimizer, loss, accuracy],
                    feed_dict={features: batch_features, labels: batch_labels})

        print( "Epoch %i of %i\t Batch %i of %i\t Loss %f\t Accuracy %f\t" % (epoch_i, epochs, batch_i, batch_count, loss_res, accuracy_res))

session.close()
