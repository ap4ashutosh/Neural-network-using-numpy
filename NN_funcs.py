# import packages
import numpy as np
np.random.seed(1)

def sigmoid(z):
    k =  1/(1 + np.exp(-z))
    return k

def sigmoid_backward(dA,cache):
    Z = cache
    s = sigmoid(Z)
    dZ = dA * s * (1-s)
    return dZ

def relu(z):
    return np.maximum(0, z)

def relu_backward(dA,cache):
    Z = cache
    dZ = np.array(dA, copy = True)
    dZ[Z<=0] = 0
    return dZ

#forward propagation functions
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1 = np.random.randn(n_h,n_x)*0.01
    W2 = np.random.randn(n_y,n_h)*0.01
    b1 = np.zeros(shape=(n_h,1))
    b2 = np.zeros(shape=(n_y,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def initialize_parameters_deep(layer_dims):
    """layer_dims keeps dimension of each layer"""
    """
     dimension of W of layer l is (nl,nl-1) //nl is no of units in layer l
     dimension of b of layer l is (nl,1)
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1,L):   # since 0 is the input layer
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])  * 0.01
        parameters['b' + str(l)] = np.zeros(shape = (layer_dims[l],1))

    return parameters

def linear_forward(A, W, b):
    """this function does forward propagation of a layer with i/p as previous layer's o/p A and it's weight W and bias b"""
    Z = np.dot(W,A) + b 
    cache = (A,W,b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """Implementation of linear activation on forward propagation"""
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev,W,b)
        activation_cache = sigmoid(Z)
        A = sigmoid(Z)

    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev,W,b)
        activation_cache = relu(Z)
        A = relu(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    np.random.seed(1)
    caches = []
    A = X
    L = len(parameters)//2  #devided by 2 to get no of layers as it containss w and b of a layer so alyer counted 2x

    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev = A_prev,W = parameters['W'+str(l)], b = parameters['b'+str(l)],activation='relu')
        caches.append(cache)
    
    # AL is activation of last layer
    AL, cache = linear_activation_forward(A_prev = A, W = parameters['W'+str(L)], b = parameters['b'+str(L)], activation='sigmoid')
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    yal = Y * np.log(AL)
    yal1 = (1-Y) * np.log(1-AL)
    cost = np.abs(np.sum(yal+yal1)/m)

    return np.squeeze(cost)

#Backpropagation functions

def linear_backward(dZ, cache):
    """This function implements linear portion of back propagation for layer 1"""
    np.random.seed(1)
    A_prev, W, b = cache
    m = A_prev.shape[-1]
    dW = np.matmul(dZ,A_prev.T)/m
    db = np.sum(dZ, axis =1, keepdims = True)/m
    dA_prev = np.matmul(W.T,dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """Implements the backprop for the linear activation layer"""
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA,activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA,activation_cache)
    dA_prev, dW, db = linear_backward(dZ = dZ, cache = linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    np.random.seed(1)
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL)) #taking derivative
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation='sigmoid')
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)): 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA'+str(l+1)], current_cache, activation = 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads

def update_parameters(params, grads, learning_rate):
    parameters = params.copy()
    L = len(parameters)//2

    for l in range(L):
        parameters['W'+str(l+1)] -= (learning_rate*grads['dW'+str(l+1)])
        parameters['b'+str(l+1)] -= (learning_rate*grads['db'+str(l+1)])
    
    return parameters

def predict(x, y, parameters):
    L = len(parameters)//2
    A = x
    for i in range(1,L):
        A = relu(np.dot(parameters['W'+str(i)],A)+parameters['b'+str(i)])
    A = sigmoid(np.dot(parameters['W'+str(L)],A)+parameters['b'+str(L)])
    return A


# def predict(w, b, X):
    
    
#     m = X.shape[1]
#     Y_prediction = np.zeros((1,m))
#     w = w.reshape(X.shape[0], 1)
#     A = sigmoid(np.dot((w.T), X) + b)
    
#     for i in range(A.shape[1]):
#         if(A[0, i] <= 0.5):
#             Y_prediction[0, i] = 0
#         else:
#             Y_prediction[0, i] = 1
    
#     assert(Y_prediction.shape == (1, m))
    
#     return Y_prediction
