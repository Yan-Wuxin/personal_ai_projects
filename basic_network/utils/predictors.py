import numpy as np
from .forward_propagation import L_model_forward

def predict(X, Y, parameters):
    m = X.shape[1]
    p = np.zeros((1, m))
    probas, _ = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    return p