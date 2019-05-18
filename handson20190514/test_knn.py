import numpy as np
import traceback

def cosine(X, Y):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    Z = 1 - np.dot(X, Y.T)
    return Z

def euclidean(X, Y):
    X = X[:, np.newaxis, :]
    X = np.hstack([X for i in range(Y.shape[0])])
    Y = Y[np.newaxis, :, :]
    Y = np.vstack([Y for i in range(X.shape[0])])
    Z = np.linalg.norm(X - Y, axis=2)
    return Z

def prediction(X1, X2, Y2, k):
    distance_matrix = euclidean(X1, X2)
    sort_index = np.argsort(distance_matrix, axis=1) 
    nearest_k = sort_index[:,:k] 
    labels = Y2[nearest_k] 
    label_num = np.sum(np.eye(10)[labels], axis=1) 
    Y = np.argmax(label_num, axis=1) 
    return Y

def test_cosine(func):
    try:
        a = np.arange(10).reshape(5,2)
        b = np.arange(10,20).reshape(5,2)
        y = func(a,b)
        y_ans = cosine(a,b)
        diff = np.linalg.norm(y-y_ans)
        assert diff < 1e-5
        print("OK")
    except Exception as e:
        print("Something wrong")
        print(e)
        traceback.print_exc()
        
def test_euclidean(func):
    try:
        a = np.arange(10).reshape(5,2)
        b = np.arange(10,20).reshape(5,2)
        y = func(a,b)
        y_ans = euclidean(a,b)
        diff = np.linalg.norm(y-y_ans)
        assert diff < 1e-5
        print("OK")
    except Exception as e:
        print("Something wrong")
        print(e)
        traceback.print_exc()
        
def test_knn(KNN):
    try:
        train_x = np.array([-6,  5,  5, -5,  5, -1,  2, -7,  9, -7,  5,  8,  1,  1,  2,  8, -3, -1,  7,  5, -9, -8,  8, -1, -7, -7, -5,  2,  3, -1]).reshape(15,2)
        train_y = np.array([4, 7, 6, 7, 8, 5, 6, 2, 5, 7, 0, 2, 7, 2, 1])
        test_x = np.array([ 0,  2,  4, -6,  5,  9,  9, -4, -1, -8]).reshape(5,2)
        knn = KNN(train_x, train_y, euclidean)
        y = knn.prediction(test_x, 3) 
        y_ans = prediction(test_x, train_x, train_y, 3)
        diff = np.linalg.norm(y-y_ans)
        assert diff < 1e-5
        print("OK")
    except Exception as e:
        print("Something wrong")  
        print(e)
        traceback.print_exc()