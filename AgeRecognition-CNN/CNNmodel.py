import tensorflow as tf

def cnn_model_fn(features, labels, mode):

    # Input Layer
    input_layer = tf.reshape(features, [-1, 64, 64, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1 (max pooling)
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Regulation with Dropout to reduce model over-fitting
    kept_activations = tf.layers.dropout(inputs=pool1,
                                         rate=0.25,
                                         training=mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=kept_activations,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2 (max_pooling)
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # second dropout layer
    kept_activations2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.25,
                                         training=mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #5
    conv5 = tf.layers.conv2d(
        inputs=kept_activations2,
        filters=64,
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #6
    conv6 = tf.layers.conv2d(
        inputs=conv5,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)


    # Pooling Layer #3 (max_pooling)
    pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2,2], strides=2)

    # Flatten tensor into a batch of vectors
    pool3_flat = tf.reshape(pool3, [-1, 8 * 8 * 64])

    # Dense Layer with 1024 neurons
    dense = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu)

    # Add a 0.5 probability that element will be dropped
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    logits = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                                mode=mode,
                                predictions=predictions)

    # Calculate Loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # Configure the Training operation with Adam
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                                loss=loss,
                                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=tf.argmax(labels, axis=1),
                                    predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
                                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)