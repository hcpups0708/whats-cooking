import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0],
                [0,1],
                [1,0],
                [1,1] ])
    
# output dataset            
y = np.array([[0,0,0,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((2,4)) -1
syn1 = 2*np.random.random((4,1)) -1

for iter in xrange(60000):

    # forward propagation
    l0 = X
    l1 = nonlin( np.dot(l0, syn0))
    l2 = nonlin( np.dot(l1, syn1))

    l2_error = y - l2

    l2_delta = l2_error * nonlin(l2,True)

    if (iter%10000) == 0:
        print "Error:", str(np.mean( np.abs(l2_error)))

    l1_error = l2_delta.dot(syn1.T)

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn1 += np.dot(l1.T, l2_delta)
    syn0 += np.dot(l0.T, l1_delta)

print "Output After Training:"
print l2
