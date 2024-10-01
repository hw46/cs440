'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

def marginal_distribution_of_word_counts(texts, word0):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the word that you want to count

    Output:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
    '''
    dict={}
    for i in texts:
        num=i.count(word0)
        dict[num]=dict.get(num,0)+1
        
    Pmarginal=np.zeros(max(dict.keys())+1)
    for i in dict:
        Pmarginal[i] = dict[i]/len(texts)
        
    return Pmarginal
    
def conditional_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word that you want to count
    word1 (str) - the second word that you want to count

    Outputs: 
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
      X1 is the number of times that word1 occurs in a document
      cX1-1 is the largest value of X0 observed in the provided texts
      CAUTION: If P(X0=x0) is zero, then P(X1=x1|X0=x0) should be np.nan.
    '''
    max0=0
    max1=1
    for i in texts:
        count=0
        for j in i:
            if j==word0:
                count+=1
        if count>max0:
            max0=count
    
    for i in texts:
        count=0
        for j in i:
            if j==word1:
                count+=1
        if count>max1:
            max1=count
            
    Mcond=np.zeros((max0+1,max1+1))
    Mword0=np.zeros(max0+1)
    
    for text in texts:
        count0=text.count(word0)
        count1=text.count(word1)
        Mcond[count0,count1]+=1
        Mword0[count0]+=1
        
    Pcond=np.zeros_like(Mcond)
    
    for i in range(max0+1):
        for j in range(max1+1):
            if Mword0[i]>0:
                Pcond[i,j]=Mcond[i,j]/Mword0[i]
            else:
                Pcond[i,j]=np.nan
            
    return Pcond

def joint_distribution_of_word_counts(Pmarginal, Pcond):
    '''
    Parameters:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0)

    Output:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
      CAUTION: if P(X0=x0) then P(X0=x0,X1=x1)=0, even if P(X1=x1|X0=x0)=np.nan.
    '''
    '''if len(Pmarginal) != Pcond.shape[0]:
        raise ValueError("Dimensions of Pmarginal and Pcond do not match.")'''
        
    Pjoint=np.zeros_like(Pcond)
    for i in range(len(Pcond)):
        for j in range(len(Pcond[0])):
            if Pmarginal[i]>0:
                Pjoint[i,j]=Pmarginal[i]*Pcond[i, j]
            else:
                Pjoint[i,j]=0
            
    return Pjoint

def mean_vector(Pjoint):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    
    Outputs:
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    '''
    x0=0
    x1=0
    for i in range(len(Pjoint)):
        for j in range(len(Pjoint[0])):
            x0+=Pjoint[i,j]*i
            x1+=Pjoint[i,j]*j
            print(x0,x1)
    mu=np.array([x0,x1])
    return mu

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''
    Sigma=np.zeros((2,2))
    for i in range(len(Pjoint)):
        for j in range(len(Pjoint[0])):
            Sigma[0,0]+=Pjoint[i,j]*(i-mu[0])*(i-mu[0])
            Sigma[1,1]+=Pjoint[i,j]*(j-mu[1])*(j-mu[1])
            Sigma[0,1]+=Pjoint[i,j]*(i-mu[0])*(j-mu[1])
            Sigma[1,0]=Sigma[0,1]
    return Sigma

def distribution_of_a_function(Pjoint, f):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       may be any hashable value (number, string, or even a tuple).

    Output:
    Pfunc (Counter) - Pfunc[z] = P(Z=z)
       Pfunc should be a collections.defaultdict or collections.Counter, 
       so that previously unobserved values of z have a default setting
       of Pfunc[z]=0.
    '''
    Pfunc=Counter()
    
    for i in range(len(Pjoint)):
        for j in range(len(Pjoint[0])):
            z=f(i,j)
            Pfunc[z]+=Pjoint[i, j]
    return Pfunc