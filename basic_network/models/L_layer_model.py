import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号“-”显示为方块的问题
from utils.initializers import initialize_parameters_deep
from utils.forward_propagation import L_model_forward
from utils.cost_functions import compute_cost
from utils.backward_propagation import L_model_backward
from utils.parameter_updaters import update_parameters

def L_layer_model(X, Y, layers_dims, learning_rate=0.0025, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print(f"第 {i} 轮迭代后的成本: {cost:.6f}")
            costs.append(cost)


    plt.plot(np.squeeze(costs))
    plt.ylabel('成本')
    plt.xlabel('迭代次数(百)')
    plt.title(f"学习率 = {learning_rate}")
    plt.show()

    # 在 L_layer_model 函数返回前添加：
    print(f"训练完成 共进行 {num_iterations} 次迭代")
    print(f"最终成本: {cost:.6f}")

    return parameters