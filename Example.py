"""
Simple file while just loads the data, creates, and trains a network
"""

import mnist_loader
import FirstNeuralNet

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = FirstNeuralNet.Network([784,30,10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
