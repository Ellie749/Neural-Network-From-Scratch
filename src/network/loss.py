

def calc_loss(y_true, y_pred, name='MSE'):


    if name == 'MSE':
        out = 0
        pairs = zip(y_true, y_pred)
        #out = sum((y_true - y_pred)**2 for y_true, y_pred in pairs)
        for y_true, y_pred in pairs:
            t = (y_true - y_pred)
            t = t*t
            out = out + t


        #print(type(out))
    elif name == 'MAE':
        pass
    
    elif name == 'RMSE':
        pass

    else:
        pass
    out.grad = 1
    return out


'''
TEST CASE
a = [1, 3]
b = [2, 1]
print(calc_loss(a, b))
'''