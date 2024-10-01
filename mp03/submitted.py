'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here
epsilon=1e-10
epsilon0=1e-5

def baseline(test, train):
    '''
    Implementation for the baseline tagger.
    input:  test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
            training data (list of sentences, with tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    sentences=[]
    seenDict=defaultdict(Counter)
    unseenDict=Counter()
    
    for i in train:
        for j,k in i:
                seenDict[j][k]+=1
                unseenDict[k]+=1
                
    for i in test:
        sentence=[]
        for j in i:
                if j in seenDict:
                        sentence.append((j,seenDict[j].most_common(1)[0][0]))
                else:
                        sentence.append((j,unseenDict.most_common(1)[0][0]))
        sentences.append(sentence)
        
    return sentences


def viterbi(test, train):
    '''
    Implementation for the viterbi tagger.
    input:  test data (list of sentences, no tags on the words)
            training data (list of sentences, with tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    sentences=[]
    globalTag=Counter()
    startTag=Counter()
    transition=defaultdict(Counter)
    emission=defaultdict(Counter)
    wordNum=0
    for sentence in train:
        wordNum+=len(sentence)
        startTag[sentence[0][1]]+=1
        for i,(j,k) in enumerate(sentence):
            globalTag[k]+=1
            emission[k][j]+=1
            if i<len(sentence)-1:
                transition[k][sentence[i+1][1]]+=1
    tagNum=len(globalTag)
    startSmooth={}
    transitionSmooth=defaultdict(dict)
    emissionSmooth=defaultdict(dict)
    for i in globalTag:
        startSmooth[i]=(startTag[i]+epsilon)/(sum(startTag.values())+epsilon*tagNum)
        for j in globalTag:
            transitionSmooth[i][j]=(transition[i][j]+epsilon)/(globalTag[i]+epsilon*tagNum)
        for k in emission[i]:
            emissionSmooth[i][k]=(emission[i][k]+epsilon)/(globalTag[i]+epsilon*wordNum)
    for i in emissionSmooth:
        emissionSmooth[i][''] = epsilon/(globalTag[i]+epsilon*wordNum)
    
    for sentence in test:
        trellis=defaultdict(lambda:defaultdict(float))
        backpointers=defaultdict(dict)
        for i in globalTag:
            trellis[i][0]=log(startSmooth.get(i,0))+log(emissionSmooth[i].get(sentence[0],emissionSmooth[i]['']))
            backpointers[i][0]=None
        for i in range(1,len(sentence)):
            for j in globalTag:
                maxP,bestP=max(
                    (trellis[k][i-1]+log(transitionSmooth[k].get(j,0))+log(emissionSmooth[j].get(sentence[i],emissionSmooth[j][''])),k)
                    for k in globalTag
                )
                trellis[j][i]=maxP
                backpointers[j][i]=bestP
        best=[]
        last=max(trellis.keys(),key=lambda i:trellis[i][len(sentence)-1])
        for i in reversed(range(len(sentence))):
            best.append(last)
            last=backpointers[last][i]
        best.reverse()
        sentences.append(list(zip(sentence,best)))
    
    return sentences


def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''