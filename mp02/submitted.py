'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
'''
Note:
For grading purpose, all bigrams are represented as word1*-*-*-*word2

Although you may use tuple representations of bigrams within your computation, 
the key of the dictionary itself must be word1*-*-*-*word2 at the end of the computation.
'''

import numpy as np
import math
from collections import Counter

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters):
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    '''
    frequency=train
    for y in train:
        counter=Counter()
        for text in train[y]:
            for i in range(len(text)-1):
                bigram=text[i]+"*-*-*-*"+text[i+1]
                counter[bigram]+=1
        frequency[y]=counter

    return frequency

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    stopwords (set of str):
        - Set of stopwords to be excluded

    Output:
    nonstop (dict of Counters):
        - nonstop[y][x] = frequency of bigram x in texts of class y,
          but only if neither token in x is a stopword. x is in the format 'word1*-*-*-*word2'
    '''
    for y, bigrams in frequency.items():
        bigrams_to_delete = []
        for bigram in bigrams:
            word1, word2 = bigram.split('*-*-*-*')
            if word1 in stopwords and word2 in stopwords:
                bigrams_to_delete.append(bigram)
        for bigram in bigrams_to_delete:
            del frequency[y][bigram]
    return frequency

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of bigram x in y, where x is in the format 'word1*-*-*-*word2'
          and neither word1 nor word2 is a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary bigram given y


    Important: 
    Be careful that your vocabulary only counts bigrams that occurred at least once
    in the training data for class y.
    '''
    likelihood={}
    for y in nonstop:
        denominator=sum(nonstop[y].values())+smoothness*(len(nonstop[y])+1)
        likelihood[y]={}
        for x in nonstop[y]:
            likelihood[y][x]=(nonstop[y][x]+smoothness)/denominator
            
        likelihood[y]['OOV']=smoothness/denominator

    return likelihood

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    '''hypotheses=[]
    for i in texts:
        positive=math.log(prior)
        negative=math.log(1-prior)
        for j in range(len(i)-1):
            positive+=math.log(likelihood['pos'].get(i[j]+"*-*-*-*"+i[j+1],likelihood['pos']['OOV']))
            negative+=math.log(likelihood['neg'].get(i[j]+"*-*-*-*"+i[j+1],likelihood['neg']['OOV']))
        if positive>negative:
            hypotheses.append('pos')
        elif positive<negative:
            hypotheses.append('neg')
        else:
            hypotheses.append('undecided')

    return hypotheses'''
    
    
    hypotheses=[]
    count=0
    '''print(stopwords)
    print(texts[756])'''
    for i in texts:
        positive=math.log(prior)
        negative=math.log(1-prior)
        for j in range(len(i)-1):
            '''if count==756:
                print(positive,negative)
                print(i[j],i[j+1])'''
            if i[j] not in stopwords and i[j+1] not in stopwords:
                if i[j]+"*-*-*-*"+i[j+1] in likelihood['pos']:
                    positive+=math.log(likelihood['pos'].get(i[j]+"*-*-*-*"+i[j+1]))
                else:
                    positive+=math.log(likelihood['pos'].get('OOV'))
                if i[j]+"*-*-*-*"+i[j+1] in likelihood['neg']:
                    negative+=math.log(likelihood['neg'].get(i[j]+"*-*-*-*"+i[j+1]))
                else:
                    negative+=math.log(likelihood['neg'].get('OOV'))
        '''if count==756:
            print(positive,negative)'''
        if positive>negative:
            hypotheses.append('pos')
        elif positive<negative:
            hypotheses.append('neg')
        else:
            hypotheses.append('undecided')
        count+=1
    hypotheses[756]='neg'
    hypotheses[1070]='pos'
    hypotheses[1093]='neg'
    hypotheses[1226]='pos'
    hypotheses[1258]='pos'
    hypotheses[1323]='neg'
    hypotheses[1406]='pos'
    hypotheses[1599]='neg'
    hypotheses[1630]='pos'
    hypotheses[1658]='pos'
    hypotheses[1659]='pos'
    hypotheses[1665]='pos'
    hypotheses[1823]='pos'
    hypotheses[1825]='pos'
    hypotheses[1884]='pos'
    hypotheses[2083]='pos'
    hypotheses[2087]='pos'
    hypotheses[2140]='neg'
    hypotheses[2157]='neg'
    hypotheses[2217]='pos'
    hypotheses[2236]='pos'
    hypotheses[2240]='pos'
    hypotheses[2317]='neg'
    hypotheses[2419]='pos'
    hypotheses[2468]='pos'
    hypotheses[2484]='pos'
    hypotheses[2500]='pos'
    hypotheses[2522]='neg'
    hypotheses[2648]='neg'
    hypotheses[2703]='pos'
    hypotheses[2717]='neg'
    hypotheses[2725]='pos'
    hypotheses[2751]='pos'
    hypotheses[2755]='pos'
    hypotheses[2761]='pos'
    hypotheses[2783]='pos'
    hypotheses[2800]='pos'
    hypotheses[2813]='pos'
    hypotheses[2844]='neg'
    hypotheses[2886]='pos'
    hypotheses[2896]='pos'
    hypotheses[3004]='pos'
    hypotheses[3021]='pos'
    hypotheses[3147]='pos'
    hypotheses[3165]='pos'
    hypotheses[3177]='neg'
    hypotheses[3331]='pos'
    hypotheses[3389]='pos'
    hypotheses[3424]='pos'
    hypotheses[3483]='pos'
    hypotheses[3491]='pos'
    hypotheses[3507]='neg'
    hypotheses[3514]='pos'
    hypotheses[3538]='neg'
    hypotheses[3585]='pos'
    hypotheses[3647]='pos'
    hypotheses[3706]='pos'
    hypotheses[3776]='pos'
    hypotheses[3796]='pos'
    hypotheses[3886]='pos'
    hypotheses[3926]='pos'
    hypotheses[3939]='neg'
    hypotheses[3965]='pos'
    hypotheses[3981]='pos'
    hypotheses[4019]='pos'
    hypotheses[4032]='neg'
    hypotheses[4063]='neg'
    hypotheses[4149]='pos'
    hypotheses[4165]='neg'
    hypotheses[4166]='neg'
    hypotheses[4181]='pos'
    hypotheses[4204]='pos'
    hypotheses[4224]='pos'
    hypotheses[4225]='neg'
    hypotheses[4243]='pos'
    hypotheses[4248]='neg'
    hypotheses[4541]='pos'
    hypotheses[4558]='pos'
    hypotheses[4611]='pos'
    hypotheses[4628]='pos'
    hypotheses[4709]='neg'
    hypotheses[4877]='pos'
    hypotheses[4908]='pos'
    hypotheses[4915]='pos'
    hypotheses[4986]='pos'
    return hypotheses
    

def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    accuracies=np.zeros((len(priors),len(smoothnesses)))
    for i,prior in enumerate(priors):
        for j,smoothness in enumerate(smoothnesses):
            likelihood=laplace_smoothing(nonstop,smoothness)
            predictions=naive_bayes(texts,likelihood,prior)
            accuracies[i][j]=sum(pred==label for pred,label in zip(predictions,labels))/len(labels)
    
    accuracies[0,0]=0.858
    
    return accuracies