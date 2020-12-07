import numpy as np
from hmm import HMM

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    
    for i, word in enumerate(unique_words.keys()):
        word2idx[word] = i
        
    for i, tag in enumerate(tags):
        tag2idx[tag] = i
    

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################
    for line in train_data:
        pi[tag2idx[line.tags[0]]] += 1
    
    if(np.sum(pi) == 0):
        pi = np.zeros(S)
    else:
        pi = pi/np.sum(pi)
    
    for line in train_data:
        for i in range(1, len(line.tags)):
            x = tag2idx[line.tags[i-1]]
            y = tag2idx[line.tags[i]]
            A[x, y] += 1
            
            
    for line in train_data:
        for tag, word in zip(line.tags, line.words):
            x = tag2idx[tag]
            y = word2idx[word]
            B[x, y] += 1
    
    A_sum = np.sum(A, axis=1)
    A[A_sum == 0, :] = 0.0
    A_sum[A_sum == 0] = 1.0
    A = A / np.expand_dims(A_sum, axis=1)
    
    B_sum = np.sum(B, axis=1)
    B[B_sum == 0, :] = 0.0
    B_sum[B_sum == 0] = 1.0
    B = B / np.expand_dims(B_sum, axis=1)
    
    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################
    S = len(tags)
    
    for line in test_data:
        for word in line.words:
            if(not word in model.obs_dict):
                model.obs_dict[word] = len(model.obs_dict)
                missing_B = np.expand_dims(np.ones(S, dtype=float)*1e-6, axis=1)
                model.B = np.concatenate((model.B, missing_B), axis=1)
                
        path = model.viterbi(line.words)
        tagging.append(path)
    
    return tagging


# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
