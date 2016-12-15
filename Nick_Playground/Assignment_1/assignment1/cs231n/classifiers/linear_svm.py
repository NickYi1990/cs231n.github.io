import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  derivative of right class and wrong class
  http://cs231n.github.io/linear-classify/#softmax
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  # Compute Loss across all data point
  for i in xrange(num_train):
    scores = X[i].dot(W) # Compute scores of point for all classes
    correct_class_score = scores[y[i]] # Compute score of the right class point
    for j in xrange(num_classes): # Update the Loss function by adding all losses
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i] # update weight of wrong classes
        dW[:,y[i]] -= X[i] # update weight of right class

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train # averaging
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W #handle regularization
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = len(y)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(X,W).T # shape is N*D . D*C= C*N
  delta = np.ones(scores.shape) # shape is C*N delta should be any positive number, but we usually choose 1
  correct_bias_index = zip(y,np.arange(0, scores.shape[1])) # the index is suit for C*N matrix, I design this for brocasting
  for i,j in correct_bias_index: # we give delta 0 for the right class index, because they should have no loss
      delta[i,j] = 0
  margins = scores - scores[y,np.arange(0, scores.shape[1])] + delta
  margins[margins<0] = 0
  loss = np.sum(margins)/num_train
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margins[margins>0] = 1 # raw data for computing weight of right class, shape is C*N
  margins[y,np.arange(0, scores.shape[1])] = -np.sum(margins, axis=0) # give weight(the sum number of class that margin > 0) to the right class, shape is C*N
  dW = np.dot(margins, X).T/num_train + reg*W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
