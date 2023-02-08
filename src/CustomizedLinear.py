import torch
import torch.nn as nn
import math


class CustomizedLinear(nn.Module):
    def __init__(self, mask, pos_weights=True, bias=True):
        """
        extended torch.nn module which mask connection.
        Arguments
        ------------------
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features, self.output_features = mask.shape
        # Transpose mask to account for multiplication with weights
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()
        # Mask should not be updated, remove gradient
        self.mask = nn.Parameter(self.mask, requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            self.register_parameter('bias', None)

        if pos_weights:
            self.reset_params_pos()
        else:
            self.reset_parameters()

    # Initialization of parameters
    def reset_parameters(self):
        """
        Initialization of parameters, sampled from U[-sqrt(nm), sqrt(nm)] where n and m are the dimensions of the
        weight matrix. The weights are then multiplied with the adjacency mask to set non-existing edges to zero
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        self.weight.data = self.weight.data * self.mask

    def reset_params_pos(self):
        """
        Same ase reset_params but only allowing for positive weights, sampling from U[0, 2*sqrt(nm)]
        :return:
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(0, 2*stdv)
        if self.bias is not None:
            self.bias.data.uniform_(0, 2*stdv)

        self.weight.data = self.weight.data * self.mask

    def forward(self, input):
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


class CustomizedLinearFunction(torch.autograd.Function):
    """
    Customized autograd function to run backward pass in pytorch
    :return: gradients with respect to input, weight, bias, mask
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())

        # add bias
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        # cache these variables for backward computation
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # weight here are already masked - see forward function
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        # M * x
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask

        # gradient for biases
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


def masked_activation(mask, pos_weights=False, activation=None, batchnorm=False):
    """
    Generating a sparse linear layer respecting the adjacency matrix given in the mask
    :param mask: np.array; adjacency matrix for connections in the linear layer
    :param pos_weights: bool; Flag to initialize parameters only with positive weights
    :param activation: str; activation function to be used, can be 'relu', 'leakyrelu', 'tanh', 'sigmoid', 'softplus'
    :param batchnorm: bool; Flag whether to used batchnorm
    :return: List(nn.module); a list of the nn.moduls in order [nn.Linear, nn.Batchnorm, nn.activation]
    """
    module = []
    feature_in, feature_out = mask.shape
    activation_pool = {'relu': nn.ReLU(),
                       'leakyrelu': nn.LeakyReLU(0.1),
                       'tanh': nn.Tanh(),
                       'sigmoid': nn.Sigmoid(),
                       'softplus': nn.Softplus()}

    module.append(CustomizedLinear(mask, pos_weights=pos_weights))

    if batchnorm:
        module.append(nn.BatchNorm1d(feature_out))

    if activation is not None:
        module.append(activation_pool[activation])

    return module
