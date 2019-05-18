import numpy as np
import traceback
# TEST
def test_sigmoid(input_class):
    try:
        func = input_class()
        x = np.arange(10).reshape(2, 5) - 5
        y = func(x)
        y_ans = 1 / (1 + np.exp(-x))
        y_back = func.backward()
        y_back_ans = y_ans * (1 -  y_ans)
        diff = np.linalg.norm(y-y_ans)
        diff_back = np.linalg.norm(y_back-y_back_ans)
        assert diff < 1e-10 and diff_back < 1e-10
        print("ok!")
    except Exception as e:
        print("Something wrong")  
        print(e)
        traceback.print_exc()
        
# TEST
def test_relu(input_class):
    try:
        func = input_class()
        x = np.arange(10).reshape(2, 5) - 5
        y = func(x)
        y_ans = x * (x > 0)
        y_back = func.backward()
        y_back_ans = 1 * (x > 0)
        diff = np.linalg.norm(y-y_ans)
        diff_back = np.linalg.norm(y_back-y_back_ans)
        assert diff < 1e-10 and diff_back < 1e-10
        print("ok!")
    except Exception as e:
        print("Something wrong")  
        print(e)
        traceback.print_exc()
        
def test_softmax(input_class):
    try:
        func = input_class()
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

def test_linear(input_class, act):
    try:
        act_func = act()
        func = input_class(5, 10, act)
        x = np.arange(10).reshape(2, 5) - 5

        np.random.seed(42)
        func.W = np.random.uniform(low=-0.08, high=0.08, size=(5, 10))
        z = func(x)

        np.random.seed(42)
        W = np.random.uniform(low=-0.08, high=0.08, size=(5, 10))
        b = np.zeros(10)
        u = np.dot(x, W) + b
        z_ans = act_func(u)

        dout = np.arange(1, 21).reshape(2, 10) - 5
        y_back = func.backward(dout)

        delta = dout * act_func.backward()
        dout_ans = np.dot(delta, W.T)
        dW_ans = np.dot(x.T, delta)
        db_ans = np.dot(np.ones(len(x)), delta)

        diff = np.linalg.norm(z - z_ans)
        diff_back = np.linalg.norm(y_back - dout_ans)
        diff_dw = np.linalg.norm(func.dW - dW_ans)
        diff_db = np.linalg.norm(func.db - db_ans)

        assert diff < 1e-10 and diff_back < 1e-10 and diff_dw < 1e-10 and diff_db < 1e-10
        print("ok!")
    except Exception as e:
        print("something wrong")
        print(e)