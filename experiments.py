import torch
import torch.nn as nn
import numpy as np

from pad2d_op import pad2d, pad2dT

class Pad2DTranspose(nn.Module):
    def __init__(self, padding):
        super(Pad2DTranspose, self).__init__()
        self.padding = padding
    
    def forward(self, x):
        pad_left, pad_right, pad_top, pad_bottom = self.padding
        
        # Remove padding from the tensor
        x = x[..., pad_top:-pad_bottom, pad_left:-pad_right]
        
        return x
    
def soft_thresholding(v, lambda_):
    return np.sign(v)*np.maximum(np.abs(v)-lambda_, 0)

def proximal_gradient_method(A, b, lambda_, alpha, num_iterations=100):
    m, n = A.shape
    x = np.zeros(n)

    for _ in range(num_iterations):
        # gradient step
        gradient = A.T@(A@x-b)
        y = x - alpha * gradient

        # proximal step
        x = soft_thresholding(y, alpha*lambda_)
    
    return x
    
if __name__ == "__main__":
    # x = torch.randn((1, 3, 64, 64)).cuda()
    # pad = 1

    # P2dT = Pad2DTranspose((pad,pad,pad,pad)).cuda()
    # x1 = P2dT(x)

    # print(x1.shape)


    # x2 = pad2dT(x, (pad,pad,pad,pad), mode='symmetric')
    # print(x2.shape)
    np.random.seed(0)
    A = np.random.randn(100, 20)
    b = np.random.randn(100)
    lambda_ = 0.1
    alpha = 0.01

    x_opt = proximal_gradient_method(A, b, lambda_, alpha)
    print("Optimized x:", x_opt)
