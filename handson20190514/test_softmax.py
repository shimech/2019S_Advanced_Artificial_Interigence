import numpy as np
import traceback

def test_softmax(func):
    try:
        x = np.array([[1,2,3,4],
                     [5,6,7,8]])
        y = func(x)
        exp_x = np.exp(x - x.max(axis=1, keepdims=True))
        y_ans = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        diff = np.linalg.norm(y-y_ans)
        assert diff < 1e-10
        print("ok!")
    except Exception as e:
        print("Something wrong")  
        print(e)
        traceback.print_exc()
        
def test_cross_entropy(func):
    try:
        y = np.array([[0.1, 0.2, 0.7]])
        t = np.array([[1, 0, 0]])
        L = func(y, t)
        L_ans = -np.mean(np.sum(t*np.log(y), axis=1))
        diff = np.linalg.norm(L-L_ans)
        assert diff < 1e-10
        print("ok!")
    except Exception as e:
        print("Something wrong")  
        print(e)
        traceback.print_exc()

def test_gradient_decent(LR):
    def gradient_decent(W, b, X, Y, T, eps):
        batchsize = X.shape[0]
        delta = Y - T 
        W = W - eps * np.dot(X.T, delta) / batchsize
        b = b - eps * np.sum(delta, axis=0) / batchsize
        return W, b
    try:
        model = LR(5,5)
        W, b = model.W.copy(), model.b.copy()
        x = np.array([[1,2,3,4,5]])
        t = np.array([[1,0,0,0,0]])
        y = np.array([[0.9, 0.1, 0, 0, 0]])
        model.gradient_decent(x, y, t, 1)
        W, b = gradient_decent(W, b, x, y, t, 1)
        W_diff = np.linalg.norm(W-model.W)
        b_diff = np.linalg.norm(b-model.b)
        assert W_diff + b_diff < 1e-10
        print("ok!")
    except Exception as e:
        print("Something wrong")  
        print(e)
        traceback.print_exc()
    
    