def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for i in range(1, L + 1):
        parameters["W" + str(i)] -= grads["dW" + str(i)] * learning_rate
        parameters["b" + str(i)] -= grads["db" + str(i)] * learning_rate
    return parameters