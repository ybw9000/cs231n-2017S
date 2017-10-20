from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    #out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    out = x.reshape(x.shape[0], -1).dot(w) + b.reshape(1, -1)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    #dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = dout.dot(w.T).reshape(*x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = dout.sum(axis = 0)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    #out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.where(x > 0, 1, 0)*dout
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        sample_mean, sample_var = np.mean(x, axis = 0), np.var(x, axis = 0)
        running_mean = momentum*running_mean + (1 - momentum)*sample_mean
        running_var = momentum*running_var + (1 - momentum)*sample_var
        x_hat = (x - sample_mean)/np.sqrt(sample_var + eps) #broadcast
        out = gamma*x_hat + beta
        #cache = (x_hat, sample_var, eps, gamma, beta)
        cache = (x, sample_mean, sample_var, x_hat, beta, gamma, eps)
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_hat = (x - running_mean)/np.sqrt(running_var + eps)
        out = gamma*x_hat + beta
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    #x_hat, sample_var, eps, gamma, beta = cache
    (x, sample_mean, sample_var, x_hat, beta, gamma, eps) = cache
    N = dout.shape[0]
    dz = gamma*dout
    dgamma = np.sum(x_hat*dout, axis = 0) #chain rule for broadcasted multiply
    dbeta = np.sum(dout, axis = 0)
    # dx = dz*(1/np.sqrt(sample_var + eps)*(1 - 1/N*(1 + x_hat**2)))
    #dsample_var = -1.0/2*dz*x_hat/(sample_var+eps) # wrong without sum
    dsample_var = np.sum(-1.0/2*dz*x_hat/(sample_var+eps), axis = 0) #it is important to do the sum here
    dsample_mean = np.sum(-1/np.sqrt(sample_var+eps)*dz, axis = 0)
    dx = 1/np.sqrt(sample_var+eps)*dz + dsample_var*2.0/N*(x-sample_mean) + 1.0/N*dsample_mean
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    (x, sample_mean, sample_var, x_hat, beta, gamma, eps) = cache
    N = dout.shape[0]
    dz = gamma*dout
    dgamma = np.sum(x_hat*dout, axis = 0) #chain rule for broadcasted multiply
    dbeta = np.sum(dout, axis = 0)
    # dx = dz*(1/np.sqrt(sample_var + eps)*(1 - 1/N*(1 + x_hat**2))) #wrong due to false diff on 1d vector
    #dsample_var = np.sum(-1.0/2*dz*x_hat/(sample_var+eps), axis = 0)
    #dsample_mean = np.sum(-1/np.sqrt(sample_var+eps)*dz, axis = 0)
    #dx = 1/np.sqrt(sample_var+eps)*dz + dsample_var*2.0/N*(x-sample_mean) + 1.0/N*dsample_mean
    dsample_var_alt = np.sum(dz*x_hat, axis = 0)
    dsample_mean_alt = np.sum(dz, axis = 0)
    #dx = 1/np.sqrt(sample_var+eps)*dz + dsample_var/N*x_hat/np.sqrt(sample_var + eps) + 1.0/N*dsample_mean/np.sqrt(sample_var + eps)
    dx = 1/np.sqrt(sample_var + eps)*(dz - 1/N*(dsample_var_alt*x_hat + dsample_mean_alt))
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = np.random.choice(np.array([0, 1]), size = x.shape, replace = True,
                           p = np.array([p, 1 - p]))
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        out = x*mask/(1 - p)
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout*mask/(1 - dropout_param['p'])
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # padding
    pad = conv_param['pad']
    x_pad = np.pad(x, pad_width = ((0,0), (0, 0), (pad, pad), (pad, pad)),
                   mode = 'constant', constant_values = 0)

    #convolution, keep in mind reshape and transpose might either
    #share the memory or do a copy.
    N, C, H, W = x_pad.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    H_out, W_out = 1 + (H - HH)//stride, 1 + (W - WW)//stride #calculate output dimension
    #reshape to get the HH*WW*C axis, can use (N, H_out, W_out, HH*WW*C) as well.
    #but use (N, H_out*W_out, HH*WW*C) for easy understanding of matrix presentation
    x_mat = np.empty(shape = (N, H_out*W_out, HH*WW*C))
    for i in range(H_out):
        for j in range(W_out):
            ij_ravel = x_pad[:, :, i*stride: i*stride + HH,
                             j*stride: j*stride + WW].reshape(N, HH*WW*C) #flatten
            x_mat[:, i*W_out + j] = ij_ravel
    w_mat = w.reshape(F, C*HH*WW).T #tronspose for doing the irregular dot below
    #1. [x_mat[0].dot(w_mat), x_mat[1].dot(w_mat)...]
    #2. transpose to move the F channel above of H and W dimension
    #3. reshape to the output dimension
    out = x_mat.dot(w_mat).transpose(0, 2, 1).reshape(N, F, H_out, W_out)

    #bias
    out = out + b.reshape(1, F, 1, 1) #reshape to do the broadcasting
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x_mat, w_mat, x_pad.shape, w.shape, b, conv_param)
    return out, cache

def stride_trick(x, stride, window_height, window_width):
    N, C, H, W = x.shape
    HH, WW = window_height, window_width
    H_out, W_out = 1 + (H - HH)//stride, 1 + (W - WW)//stride
    x_strided_shape = (N, H_out, W_out, C, HH, WW)
    x_strides = x.strides
    #all axis are just spaced by the given strides
    x_trick_strides = (x_strides[0], ) + \
                      (x_strides[2]*stride, x_strides[-1]*stride) + \
                      (x_strides[1], x_strides[2], x_strides[-1])
    return np.lib.stride_tricks.as_strided(x, x_strided_shape, x_trick_strides)

def conv_forward_stride_trick(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # padding
    pad = conv_param['pad']
    x_pad = np.pad(x, pad_width = ((0,0), (0, 0), (pad, pad), (pad, pad)),
                   mode = 'constant', constant_values = 0)

    #convolution, keep in mind reshape and transpose is just reordering,
    #all created variables with these two methods share the memory.
    N, C, H, W = x_pad.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    H_out, W_out = 1 + (H - HH)//stride, 1 + (W - WW)//stride #calculate output dimension
    x_strided = stride_trick(x_pad, stride, HH, WW) #build stride trick tensor
    assert x_strided.shape == (N, H_out, W_out, C, HH, WW), print('shape mismatch')
    #adapts to the matrix presentation 0f (N, H_out*W_out, C*HH*WW)
    x_mat = x_strided.reshape(N, H_out*W_out, C*HH*WW) #reshape still consums time
    w_mat = w.reshape(F, C*HH*WW).T #tronspose for doing the irregular dot below
    #1. [x_mat[0].dot(w_mat), x_mat[1].dot(w_mat)...]
    #2. transpose to move the F channel above of H and W dimension
    #3. reshape to the output dimension
    out = x_mat.dot(w_mat).transpose(0, 2, 1).reshape(N, F, H_out, W_out)

    #bias
    out = out + b.reshape(1, F, 1, 1) #reshape to do the broadcasting
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x_mat, w_mat, x_pad.shape, w.shape, b, conv_param)
    return out, cache

def conv_forward_tensordot(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # padding
    pad = conv_param['pad']
    x_pad = np.pad(x, pad_width = ((0,0), (0, 0), (pad, pad), (pad, pad)),
                   mode = 'constant', constant_values = 0)

    #convolution, keep in mind reshape and transpose is just reordering,
    #all created variables with these two methods share the memory.
    N, C, H, W = x_pad.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    H_out, W_out = 1 + (H - HH)//stride, 1 + (W - WW)//stride #calculate output dimension
    x_strided = stride_trick(x_pad, stride, HH, WW) #build stride trick tensor
    assert x_strided.shape == (N, H_out, W_out, C, HH, WW), print('shape mismatch')
    #tensor dot along C, HH, WW axis
    out = np.tensordot(x_strided, w, axes = ([3, 4, 5], [1, 2, 3])).transpose(0, 3, 1, 2)
    #bias
    out = out + b.reshape(1, F, 1, 1) #reshape to do the broadcasting
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x_strided, x_pad.shape, w, b, conv_param)
    return out, cache

def conv_forward_tensormultiply(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # padding
    pad = conv_param['pad']
    x_pad = np.pad(x, pad_width = ((0,0), (0, 0), (pad, pad), (pad, pad)),
                   mode = 'constant', constant_values = 0)

    #convolution, keep in mind reshape and transpose is just reordering,
    #all created variables with these two methods share the memory.
    N, C, H, W = x_pad.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    H_out, W_out = 1 + (H - HH)//stride, 1 + (W - WW)//stride #calculate output dimension
    x_strided = stride_trick(x_pad, stride, HH, WW) #build stride trick tensor
    assert x_strided.shape == (N, H_out, W_out, C, HH, WW), print('shape mismatch')
    #reshape to the output shape (N, F, H_out, W_out, C, HH, WW)
    x_reshape = x_strided.reshape(N, 1, H_out, W_out, C, HH, WW) #reshape still consums time
    w_reshape = w.reshape(1, F, 1, 1, C, HH, WW)
    # element wise multiply and sum over the last three axis
    out = np.sum(x_reshape*w_reshape, axis = (4, 5, 6))
    #bias
    out += b.reshape(1, F, 1, 1) #reshape to do the broadcasting
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x_reshape, w_reshape, x_pad.shape, w.shape, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    N, F, H_out, W_out = dout.shape
    x_mat, w_mat, x_pad_shape, w_shape, b, conv_param = cache

    #calculate dw which is a sum of x_mat[i].T.dot(dout_mat[i])
    #reverse the transpose and reshape process to get x_mat.dot(w_mat)
    dout_mat = dout.reshape(N, F, H_out*W_out).transpose(0, 2, 1)
    dw_mat = np.zeros(shape = w_mat.shape)
    for i in range(len(dout_mat)):
        dw_mat += x_mat[i].T.dot(dout_mat[i]) #calculus of matrix dot
    dw = dw_mat.T.reshape(w_shape) #adapts to w shape

    #collapse db to 1d array
    db = dout.sum(axis = (0, 2, 3))

    #calculate dx which is an overlay of the mask (w) window: reverse conv
    p, stride = conv_param['pad'], conv_param['stride']
    C, HH, WW = w_shape[1], w_shape[2], w_shape[3] #w shape
    dx_mat = dout_mat.dot(w_mat.T) #[dout_mat[0].dot(w_mat.T), dout_mat[1].dot(w_mat.T)...]: calculus of matrix dot
    dx_mat_reshape = dx_mat.reshape(N, H_out, W_out, C, HH, WW) #for convient indexing
    dx_pad = np.zeros(shape = x_pad_shape)
    #overlapping the striding mask window: reverse conv, speed mainly limited here
    for i in range(H_out):
        for j in range(W_out):
            dx_pad[:, :, i*stride: i*stride + HH, j*stride: j*stride + WW] += \
            dx_mat_reshape[:, i, j]
            #dx_mat_reshape[:, i, j].reshape(N, C, HH, WW) #protection
    dx = dx_pad[:, :, p:-p, p:-p] # get rid of padding
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def conv_backward_tensordot(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    N, F, H_out, W_out = dout.shape
    x_strided, x_pad_shape, w, b, conv_param = cache

    #grad with respect to dw
    dw = np.tensordot(dout, x_strided, axes = ([0, 2, 3], [0, 1, 2]))

    #collapse db to 1d array
    db = dout.sum(axis = (0, 2, 3))

    #calculate dx which is an overlay of the mask (w) window: reverse conv
    p, stride = conv_param['pad'], conv_param['stride']
    C, HH, WW = w.shape[1], w.shape[2], w.shape[3] #w shape

    #grad with respect to dx
    dx_reshape = np.tensordot(dout, w, axes = (1, 0))

    dx_pad = np.zeros(shape = x_pad_shape)
    #overlapping the striding mask window: reverse conv, speed mainly limited here
    for i in range(H_out):
        for j in range(W_out):
            dx_pad[:, :, i*stride: i*stride + HH, j*stride: j*stride + WW] += \
            dx_reshape[:, i, j]
            #dx_mat_reshape[:, i, j].reshape(N, C, HH, WW) #protection
    dx = dx_pad[:, :, p:-p, p:-p] # get rid of padding
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def conv_backward_tensormultiply(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    N, F, H_out, W_out = dout.shape
    x_reshape, w_reshape, x_pad_shape, w_shape, b, conv_param = cache

    #reshape to adapts x_reshape and w_reshape
    dout_reshape = dout.reshape(N, F, H_out, W_out, 1, 1, 1)

    #grad with respect to dw
    dw = np.sum(x_reshape*dout_reshape, axis = (0, 2, 3))

    #collapse db to 1d array
    db = dout.sum(axis = (0, 2, 3))

    #calculate dx which is an overlay of the mask (w) window: reverse conv
    p, stride = conv_param['pad'], conv_param['stride']
    C, HH, WW = w_shape[1], w_shape[2], w_shape[3] #w shape

    #grad with respect to dx
    dx_reshape = np.sum(w_reshape*dout_reshape, axis = 1)

    dx_pad = np.zeros(shape = x_pad_shape)
    #overlapping the striding mask window: reverse conv, speed mainly limited here
    for i in range(H_out):
        for j in range(W_out):
            dx_pad[:, :, i*stride: i*stride + HH, j*stride: j*stride + WW] += \
            dx_reshape[:, i, j]
            #dx_mat_reshape[:, i, j].reshape(N, C, HH, WW) #protection
    dx = dx_pad[:, :, p:-p, p:-p] # get rid of padding
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    #calculate output shape
    H_out, W_out = 1 + (H - HH)//stride, 1 + (W - WW)//stride
    # build mask for backprop
    # Can use (N, C, H_out*W_out, HH*WW) similar to conv layers
    # but use (N, C, H_out, W_out, HH, WW) for simplicity since there is no
    # matrix multiply as in the conv layers
    mask = np.empty(shape = (N, C, H_out, W_out, HH, WW))
    #build out
    out = np.empty(shape = (N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            window = x[:, :, i*stride: i*stride + HH, j*stride: j*stride + WW]
            maxes = np.max(window, axis = (2, 3)) #find maximum in the window
            out[:, :, i, j] = maxes
            mask[:, :, i, j] = (x[:, :, i*stride: i*stride + HH,
                         j*stride: j*stride + WW] == maxes.reshape(N, C, 1, 1))
                         #reshape for the right broadcasting
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (mask, x.shape, pool_param)
    return out, cache

def stride_trick_pool(x, stride, window_height, window_width):
    N, C, H, W = x.shape
    HH, WW = window_height, window_width
    H_out, W_out = 1 + (H - HH)//stride, 1 + (W - WW)//stride
    x_strided_shape = (N, C, H_out, W_out, HH, WW)
    x_strides = x.strides
    x_trick_strides = x_strides[:2] + \
                      (x_strides[2]*stride, x_strides[-1]*stride) + \
                      (x_strides[2], x_strides[-1])
    return np.lib.stride_tricks.as_strided(x, x_strided_shape, x_trick_strides)

def max_pool_forward_stride_trick(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    #calculate output shape
    H_out, W_out = 1 + (H - HH)//stride, 1 + (W - WW)//stride
    # build stride trick tensor
    x_stride = stride_trick_pool(x, stride, HH, WW)
    assert x_stride.shape == (N, C, H_out, W_out, HH, WW), print('shape mismatch')
    # find the maximum
    out = np.max(x_stride, axis = (4, 5))
    # build mask
    mask = (x_stride == out.reshape(N, C, H_out, W_out, 1, 1))
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (mask, x.shape, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    mask, x_shape, pool_param = cache
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    N, C = x_shape[0], x_shape[1]
    H_out, W_out = dout.shape[2], dout.shape[3]
    #initialize dx
    dx = np.zeros(shape = x_shape)
    for i in range(H_out):
        for j in range(W_out):
            #sum over overlapping windows
            dx[:, :, i*stride: i*stride + HH, j*stride: j*stride + WW] += \
            (dout[:, :, i, j].reshape(N, C, 1, 1)*mask[:, :, i, j])
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = x.shape
    x_spatial_batch = x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1])
    out_2d, cache = batchnorm_forward(x_spatial_batch, gamma, beta, bn_param)
    out = out_2d.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    dout_spatial_batch = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_2d, dgamma, dbeta = batchnorm_backward_alt(dout_spatial_batch, cache)
    dx = dx_2d.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
