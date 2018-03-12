import tensorflow as tf
import shutil
import argparse
import CNNmodel
import imageData

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int,
                                default=3000,)
parser.add_argument('-trainCSV', type=str, default='train.csv',
                                help='Csv file containing image file names and their corresponding labels used for training')
parser.add_argument('-trainImagesDir', type=str, default= 'Train',
                                help='Directory containing images listed in the csv file used for training')
parser.add_argument('-testCSV', type=str, default='test.csv',
                                help='Csv file containing image file names used for training')
parser.add_argument('-testImagesDir', type=str, default='Test',
                                help='Directory containing images listed in the csv file used for predictions')
parser.add_argument('--batchSize', type=int,
                                default=100,
                                help='Size of batches used at training')
parser.add_argument('--modelDir', type=str,
                                help='Directory to save the model',
                                default='ageRecognition_convnet_model')


def main(argv):

    args = parser.parse_args(argv[1:])

    # Clean up the model directory
    shutil.rmtree(args.modelDir, ignore_errors=True)

    # get dataframes for images and labels for training
    train, validate = imageData.train_and_eval_sets(args.trainCSV, args.trainImagesDir)

    # Create the Estimator
    classifier = tf.estimator.Estimator(
            model_fn=CNNmodel.cnn_model_fn, model_dir=args.modelDir)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    classifier.train(
            input_fn=lambda: imageData.input_fn(train, None, True, args.batchSize),
            steps=args.steps,
            hooks=[logging_hook])

    # Evaluate the model and print results
    eval_results = classifier.evaluate(
            input_fn=lambda: imageData.input_fn(validate, 1, False, args.batchSize))
    print(eval_results)

    #get Images Dataframe
    test = imageData.prediction_set(args.testCSV, args.testImagesDir)

    classifier = tf.estimator.Estimator(
            model_fn=CNNmodel.cnn_model_fn, model_dir=args.modelDir)


    # predict on test data
    predictions = None
    try:
        predictions = classifier.predict(
                input_fn=lambda: imageData.predict_input_fn(test, args.batchSize))
    except ValueError as error:
        print('Error while trying to predict: '+error.message)


    if predictions is not None:
        # write predicted labels to test csv
        imageData.store_predicted_labels(test, predictions, args.testCSV)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)