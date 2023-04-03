# Coursera - Deep learning specialization

Scripts from Coursera Deep learning specialization delivered by Andrew Ng

The specialization goes through core concepts of deep learning, building from the basic forward and backward propagation, to applications in CNN, ResNet and RNN.

1. Basics of neural network:

Build neural networks from scratch, through forward and backward propagation

Simple neural network with logistic regression to classify images

2. Hyperparameter tuning, regularization and optimization:

* How to improve model performance by optimizing initialization of weights, regularization to avoid overfitting, and. minibatch gradient descent
* Optimization algorithms like Momentum and Adam
* Gradient checking to make sure model is running correctly

3. Convolutional and Residual neural networks:

* Building a CNN from scratch, including building convolutional and pooling layers

* Creating a full CNN model using tensorflow: create placeholder, initialize parameters, forward propagate, compute cost, and create optimizer

* Neural style transfer: art generation
Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. This script uses VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low level features (at the shallower layers) and high level features (at the deeper layers).

Using tensorflow, we pass a content and a style image through the VGG-19, compute the total cost function based on the content and style cost functions, and optimize the total cost funnction using Adam optimizer. 

* Car detection for autonomous driving
"You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real-time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

We run an input image through a CNN which outputs a 19x19x5x85 dimensional volume.
The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
We filter through all the boxes using non-max suppression. Specifically:
* Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
* Intersection over Union (IoU) thresholding to eliminate overlapping boxes

* Face recognition:

FaceNet is based on Inception netowork and learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.

The triplet loss is an effective loss function for training a neural network to learn an encoding of a face image.
The same encoding can be used for verification and recognition. Measuring distances between two images' encodings allows us to determine whether they are pictures of the same person.

* Residual network:

Using Keras, we build a very deep convolutional networks, using Residual Networks (ResNets). In theory, very deep networks can represent very complex functions; but in practice, they are hard to train. Residual Networks, introduced by He et al., allow you to train much deeper networks than were previously practically feasible.

In ResNets, a "shortcut" or a "skip connection" allows the model to skip layers. Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are same or different. I implemented both of them: the "identity block" and the "convolutional block" and built the ResNet50 model based on these blocks.

4. Sequence models:

Recurrent Neural Networks (RNN) are very effective for Natural Language Processing and other sequence tasks because they have "memory". They can read inputs 
 (such as words) one at a time, and remember some information/context through the hidden layer activations that get passed from one time-step to the next. This allows a unidirectional RNN to take information from the past to process later inputs. A bidirectional RNN can take context from both the past and the future.

* Neural Machine Translation (NMT) model

Translate human-readable dates ("25th of June, 2009") into machine-readable dates ("2009-06-25") using an attention model, one of the most sophisticated sequence-to-sequence models. Both pre-attention and post=attention Bi-LSTM are implemented. 

* Generating new dinosaur names using RNN and Long Short-Term Memory (LSTM) network

* Emojifying sentences

The input of the model is a string corresponding to a sentence (e.g. "I love you).
The output will be a probability vector of shape (1,5), (there are 5 emojis to choose from).
The (1,5) probability vector is passed to an argmax layer, which extracts the index of the emoji with the highest probability.

* Operations word vectors:
Cosine similarity
Debiasing word vectors

* Trigger word detection

Trigger word detection is the technology that allows devices like Amazon Alexa, Google Home, Apple Siri, and Baidu DuerOS to wake up upon hearing a certain word.
For this exercise, our trigger word will be "Activate." Every time it hears you say "activate," it will make a "chiming" sound.

Used helper functions to create the model based on 1-D convolutional layers, GRU layers, and dense layers.
