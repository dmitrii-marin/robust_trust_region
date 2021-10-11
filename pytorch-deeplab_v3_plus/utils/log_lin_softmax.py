import torch
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F


class LogLinSoftmax(Function):
    # computes log(a + b * s_ijkl) where s_ijkl is softmax of the input

    @staticmethod
    def forward(ctx, a, b, logits, dim):
        ctx.dim, ctx.a, ctx.b = dim, a, b

        with torch.no_grad():
            s = F.softmax(logits, dim=dim)
            ctx.save_for_backward(s)
            result = torch.log(a + b * s)

        return Variable(result, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        s, = ctx.saved_tensors
        m = ctx.b * s
        m = m / (ctx.a + m)
        gimi = torch.sum(grad_output * m, dim=ctx.dim, keepdim=True)
        grad_logits = m * grad_output - s * gimi
        return None, None, grad_logits, None


def log_lin_softmax(a, b, input, dim):
    # return input.log_softmax(dim)
    return LogLinSoftmax.apply(a, b, input, dim)
    #return torch.log(a + b * input.softmax(dim))


if __name__ == '__main__':
    logits = (torch.rand(2,3,4,5) - 0.5) * 100

    input1 = logits.clone().requires_grad_(True)
    out1 = log_lin_softmax(0, 1, input1, 1)
    out1[:,0].sum().backward()

    input2 = logits.clone().requires_grad_(True)
    out2 = F.log_softmax(input2, dim=1)
    out2[:,0].sum().backward()

    input3 = logits.clone().requires_grad_(True)
    out3 = torch.log(F.softmax(input3, dim=1))
    out3[:,0].sum().backward()

    input4 = logits.clone().double().requires_grad_(True)
    out4 = log_lin_softmax(0, 1, input4, 1)
    out4[:,0].sum().backward()

    input5 = logits.clone().double().requires_grad_(True)
    out5 = F.log_softmax(input5, dim=1)
    out5[:,0].sum().backward()

    print("log_lin_softmax", input1.dtype, torch.norm(input1.grad - input5.grad))
    print("    log_softmax", input2.dtype, torch.norm(input2.grad - input5.grad))
    print("log  *  softmax", input3.dtype, torch.norm(input3.grad - input5.grad))
    print("log_lin_softmax", input4.dtype, torch.norm(input4.grad - input5.grad))

    print('---------------------------')

    for a in [0, 1e-9, 1e-5, 1e-4, 1e-2, 0.1, 0.2, 0.3, 0.4, 0.5]:
        b = 1 - a * 2
        input6 = logits.clone().double().requires_grad_(True)
        out6 = torch.log(a + b * input6.softmax(1))
        out6[:,0].sum().backward()

        input7 = logits.clone().double().requires_grad_(True)
        out7 = log_lin_softmax(a, b, input7, dim=1)
        out7[:,0].sum().backward()

        print("log_lin_softmax_ab", input7.dtype, torch.norm(input7.grad - input6.grad))
