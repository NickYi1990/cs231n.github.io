import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """

  #
  # #############################################################################
  # # : Compute the softmax loss and its gradient using explicit loops.     #
  # # Store the loss in loss and the gradient in dW. If you are not careful     #
  # # here, it is easy to run into numeric instability. Don't forget the        #
  # # regularization!                                                           #
  # #############################################################################
  loss = 0.0
  dW = np.zeros_like(W) # shape is (D,C)
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in xrange(num_train):
      scores = np.dot(X[i],W) # shape is 1*C
      scores -= np.max(scores) #handle numeric stability
      cache_denominator = np.sum(np.exp(scores)) # store a cache, saving compute time
      for j in xrange(num_class):
          if j == y[i]:
              dW[:,j] += ((np.exp(scores[j]) / cache_denominator) - 1.0)*X[i] # compute derivative for the right class's weight
          else:
              dW[:,j] += (np.exp(scores[j]) / cache_denominator)*X[i]  #compute derivative for the wrong class's weight
          #fucking checkpint..... i forget to write else, waste my whole afternoon!!!!!!!!!!!!!!!11
        #   print "deri=",(np.exp(scores[j]) / cache_denominator)-1
        #   print "xi=",np.sum(X[i])
        #   print np.sum(dW[:,j])
      loss += -np.log(np.exp(scores[y[i]]) / cache_denominator) #compute loss

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg*W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_dim = W.shape[0]
  num_train = X.shape[0]
  num_class = W.shape[1]
  # Think about create a matrix with all scores!!!!!!!!!!!!
  correct_class_weight = W.T[y]
  print np.sum(np.multiply(X,correct_class_weight), axis=1).shape
  print np.sum(np.dot(X,W), axis=1).shape
  loss = np.sum(-np.log(
                        np.divide(
                                np.exp(np.sum(np.multiply(X,correct_class_weight), axis=1)),
                                np.sum(np.exp(np.dot(X,W)), axis=1)))
                        )

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
