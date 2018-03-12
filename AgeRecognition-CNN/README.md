We try to solve the age detection problem described in  <a href="https://www.w3schools.com">here</a>. 

The model's architecture is a Convolutional Neural Network (to see the graph open graph.png image)
Current accuracy is at 0.7616033755

Note: Despite the use of dropout layer, in order to prevent over-fitting, after every max pooling layer with dropout prob 0,25
we still observe some signs that the model indeed tents to overfit (TODO: fix this in later version). Maybe a viable solution
is to add data augentation to artificially increase the size of the training set because it is rather small

DATA PREPROCESSING:
Because data images do not have the same size, resolution and saturation, we resize images to a fixed size (64, 64), 
convert then to grayscale and normalize in order to feed to the model

