import pickle
import math
import nltk
import glob 
import collections

# Inverse Document Frequency Formula
# param: numFiles - number of files used to create the corresponding fd
# param: model - loaded model dictionary
# param: fd - mapping between tokens and how many files they occur in
# param: token - token in the current file used to determine weight
# returns: log(# of documents / # of documents where token appears)
def idf(numFiles, model, fd, token):
    if token in model[fd]:
        return math.log(numFiles/model[fd][token])
    else:
        return 0

# Term Frequency computed with a tokens raw count
# Learned in class
# param: token - current token
# param: token - list of tokens in the current file
# returns: the number of times the current token appears in the file
def tf_rawFreq(tokens, token):
    return tokens.count(token)

# Term Frequency computed with the tokens raw count and with respect to the number of tokens in the file
# Studied from: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
# param: token - current token
# param: token - list of tokens in the current file
# returns: an average between the number of times the current token appears in the file and the total number of tokens in the file
def tf_lengthFreq(tokens,token):
    return tokens.count(token)/len(tokens)

# Term Frequency computed with a boolean value dependent on if a token is present in a file
# Studied from: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
# param: token - current token
# param: token - list of tokens in the current file
# returns: 1 if the token is present in the file, 0 otherwise
def tf_boolFreq(tokens, token):
    return (token in tokens)

# Term Frequency computed with the logarithmic scaling of a tokens raw count
# Studied from https://en.wikipedia.org/wiki/Tf%E2%80%93idf
# param: token - current token
# param: token - list of tokens in the current file
# returns: log(1 + the number of times the current token appears in the file)
def tf_logFreq(tokens, token):
    return math.log(1 + tokens.count(token))

# Term Frequency computed with the augmented frequency developed from minimizing the effect of having tokens reappear in larger documents
# Studied from: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
# param: token - current token
# param: token - list of tokens in the current file
# returns: Augmented Frequency of a token
def tf_augFreq(tokens, token):
    # Assign a count to each token in the file
    counter = collections.Counter(tokens)
    # Finding the token with the highest count
    maxPair = counter.most_common(1)[0]
    # Getting the highest count from the pair : (token, count)
    _, maxValue = maxPair
    # returning 0.5 + (0.5 * (the number of times the current token appears in the file / the token with the highest recurrence))
    return 0.5 + (0.5 * (tokens.count(token)/maxValue))
    
#Function used to compute the conditional probabilities for a class
# param: model - loaded model dictionary
# param: prior - prior probability of the respective pos/neg class 
# param: fd - file descriptor for the pos/neg class ... the prior and fd variables will be for the same class
# param: filetokens - list of tokens from a documents 
# param: termDocs - mapping between tokens and how many files they occur in
# param: numFiles - number of files used to create the corresponding fd
# param: allTypes - combination of both file descriptors
# returns: P(C) * P(D|C) -> Probability of the prior class multiplied by the probability of the document with respect to the class
def classifier(model, prior, fd, fileTokens, termDocs, numFiles, allTypes):
    cp = 0
    #Iterating through all the tokens to compute P(doc|class)
    for token in fileTokens: 
        # If the token is present
        if token in model[fd]:
            # Compute the term Frequency
            tf = tf_augFreq(fileTokens, token)
            # Computing the tf-idf balanced weight
            weight = (tf * idf(numFiles, model, termDocs, token))
            # Adding the weight to influence the conditional probability 
            token_class = (model[fd][token] + weight) / (model[fd].N() + allTypes)
        else:
            # If the token is not present in the file, perform add one smoothing
            token_class = 1 / (model[fd].N() + allTypes)
        #Accumulating the logs ... log(A*B) = log(A) + log(B)
        cp = math.log(token_class) + cp   
    return (math.log(prior) + (cp))

def main():
    # Get list of files in the test set
    test_files = sorted(glob.glob('/home/hrrathod/project/test/*.txt'))
    # restore model from airline.nb
    model = pickle.load(open('/home/hrrathod/project/naive_bayes/airline.nb', 'rb')) 
    # Total number of types in all reviews
    vocab_size = model['pos_fd'].B() + model['neu_fd'].B() + model['neg_fd'].B()

    #Will be passed into the classifier function
    posDesc = 'pos_fd'
    neuDesc = 'neu_fd'
    negDesc = 'neg_fd'

    # Priors (log)
    positive_prior = (model['pos_n']) / (model['pos_n'] + model['neu_n'] + model['neg_n'])
    neutral_prior = (model['neu_n']) / (model['pos_n'] + model['neu_n'] + model['neg_n'])
    negative_prior = (model['neg_n']) / (model['pos_n'] + model['neu_n'] + model['neg_n'])

    # Open the output file
    output_file = open('/home/hrrathod/project/tfidf/tfidf_augFreq_predictions.txt', 'w')

    # Go through all the files in the test set
    for file_name in test_files:
        # Open the current file
        current_file = open(file_name, 'r')
        # Read the file 
        text = current_file.read()
        # Tokenize the text in the current file
        tokens = nltk.word_tokenize(text)

        # Computing the probabilities for all sentiments
        posRes = classifier(model, positive_prior, posDesc, tokens, 'pos_docs', model['pos_n'], vocab_size)
        neuRes = classifier(model, neutral_prior, neuDesc, tokens, 'neu_docs', model['neu_n'], vocab_size)
        negRes = classifier(model, negative_prior, negDesc, tokens, 'neg_docs', model['neg_n'], vocab_size)
       
        # Assigning the highest probability as the predicted label
        highestProb = max(posRes, neuRes, negRes)
        if highestProb == posRes:
            print(current_file.name, '\tpos', file = output_file) # printing results 
        elif highestProb == neuRes:
            print(current_file.name, '\tneu', file = output_file) # printing results 
        else:
            print(current_file.name, '\tneg', file = output_file) # printing results     
                      
    # closing the output file
    output_file.close()

    

    
    



if __name__ == "__main__":  
    main()