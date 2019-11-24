import numpy as np
from Helper import ActivationFunction


def element_wise_operation(input, weight):
    """
    Using for element wise operation of 2 matrix (input and weight)
    :param input: a matrix (kernel_height, kernel_width, previous_depth), example: 3x3x1
    :param weight: a matrix (kernel_height, kernel_width, previous_depth, this_depth), example: 3x3x1x32
    :return: result of input x weight (x is element wise operation)
    """
    tmp_input = []
    trans_input = np.transpose(input)  # previous_depth x width_kernel x height_kernel
    for index in range(weight.shape[-1]):  # this_depth
        tmp_input.append(trans_input)
    new_input = np.transpose(tmp_input)  # kernel_height x kernel_width x previous_depth x this_depth
    output = new_input * weight  # kernel_height, kernel_width, previous_depth, this_depth
    output = np.sum(output, axis=(0, 1, 2))  # [this depth]
    return output


def get_sub_input(input, i, j, sub_size):
    """
    Use to get the sub input
    :param input: a matrix
    :param i: the start point
    :param j: the end point
    :param sub_size: the size of this sub input
    :return: the sub input
    """
    i_end = i+sub_size[0]
    j_end = j+sub_size[1]
    sub_input = input[i:i_end, j:j_end]
    return sub_input


def convolution_operation(input, weight, kernel_size, strides):
    """
    Use to calculate the convolution of a input data point in conv2d layer
    :param input: a data point with 3-dimensions (height, width, previous depth), example: 28x28x1
    :param weight: a 4-dimensions matrix (kernel_height, kernel_width, previous depth, this depth) example: 3x3x1x32
    :param kernel_size: size of kernel
    :param strides: step size
    :return: a output (height, width, this depth) of this convolution operation
    """
    output = []  # Height x Width x depth
    i = 0
    while i <= len(input)-kernel_size[0]:
        j = 0
        output_vector = []  # Width x depth
        while j <= len(input[0])-kernel_size[1]:
            sub_input = get_sub_input(input, i, j, kernel_size)
            tmp_output_vector = element_wise_operation(sub_input, weight)
            output_vector.append(tmp_output_vector)
            j += strides[1]
        output.append(output_vector)
        i += strides[0]
    return np.array(output)


def conv2d(input, layer, weight, bias):
    """
    Use to calculate the output for the convolution2d layer
    :param input: a 4-dimensions matrix
    :param layer: the information of convolution2d layer
    :param weight: a 4-dimensions matrix (n x height x width x previous depth x this depth)
    :param bias: a matrix 1-dimension (depth) uses after convolution operation
    :return: a 4-dimensions output of this convolution layer
    """
    predicted_output = []
    kernel_size = layer.kernel_size
    strides = layer.strides
    if layer.padding == 'same':
        input = np.pad(input, [(0, 0), (1, 1), (1, 1), (0, 0)], mode='constant')  # padding = 1 for second and third dimension with value = 0
    for element in input:  # element is an image with height x width x depth_of_image
        sub_output = convolution_operation(element, weight, kernel_size, strides)  # 28x28x32 height x width x depth of kernel
        predicted_output.append(sub_output)
    predicted_output += bias
    return np.array(predicted_output)


def handle_data_point(input, pool_size, strides):
    """
    Use to handle each data point of input in the max_pooling2d function
    :param input: a matrix that is a data point of overall input
    :param pool_size: size of pooling
    :param strides: step size
    :return: corresponding output of this input data point
    """
    output = []  # Height x Width x depth
    i = 0
    while i < input.shape[0]:  # Height
        j = 0
        output_vector = []  # Width x depth
        while j < input.shape[1]:
            sub_input = get_sub_input(input, i, j, pool_size)
            tmp_output_vector = np.max(sub_input, axis=(0, 1))
            output_vector.append(tmp_output_vector)
            j += strides[1]
        output.append(output_vector)
        i += strides[0]
    return np.array(output)


def max_pooling2d(input, pool_size, strides):
    """
    Use to reduce the size of a matrix
    :param input: a 4-dimension matrix
    :param pool_size: size of pooling, example (2,2)
    :param strides: step size, example (2,2)
    :return: a corresponding output after reducing the size of input matrix
    """
    output = []
    for element in input:
        tmp_output = handle_data_point(element, pool_size, strides)
        output.append(tmp_output)
    return np.array(output)


def flatten(input):
    """
    Use to flatten a matrix input
    :param input: a 4-dimensions matrix
    :return: output with shares are n x -1
    """
    output = np.reshape(input, (input.shape[0], -1))
    return output


def dense(input, weight, bias):
    """
    use to handle the dense layers
    :param input: a matrix input with shapes are n x the num_of_nodes of previous layer
    :param weight: shapes are the num_of_nodes of previous layer x the num_of_nodes of this dense layer
    :param bias: shape is the num_of_nodes of this dense layer
    :return: output with shapes are n x the num_of_nodes of this dense layer
    """
    output = input.dot(weight) + bias
    return output


def handle_layer(input, layer, weights, index_of_weight):
    """
    Use to handle each layer of model
    :param input: corresponding input of this layer
    :param layer: a layer of model
    :param weights: corresponding weights of this layer
    :param index_of_weight: indicate the weight index of this layer
    :return: predicted output of this layer
    """
    predicted_output = None
    if 'conv2d' in layer.name:
        predicted_output = conv2d(input, layer, weights[index_of_weight], weights[index_of_weight+1])
        activation = layer.output.op.node_def.input[0].split('/')[-1]  # get activation function name
        predicted_output = ActivationFunction.dispatcher[activation](predicted_output)
        index_of_weight += 2  # used 2 weights
    elif 'max_pooling2d' in layer.name:
        predicted_output = max_pooling2d(input, layer.pool_size, layer.strides)
    elif 'flatten' in layer.name:
        predicted_output = flatten(input)
    elif 'dense' in layer.name:
        predicted_output = dense(input, weights[index_of_weight], weights[index_of_weight+1])
        activation = layer.output.op.node_def.input[0].split('/')[-1]  # get activation function name
        predicted_output = ActivationFunction.dispatcher[activation](predicted_output)
        index_of_weight += 2  # used 2 weights
    return predicted_output, index_of_weight


def predict(input, layers, weights):
    """
    use to predict output
    :param input: a matrix with 4-dimensions n x height x width x depth
    :param layers: layers of model
    :param weights: weights of model after training
    :return: predicted the output of model, shape = (n, 10)
    """
    predicted_output = input
    index_of_weight = 0
    for index in range(len(layers)):
        predicted_output, index_of_weight = handle_layer(predicted_output, layers[index], weights, index_of_weight)
    return predicted_output
