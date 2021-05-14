import pickle
import glob
import nltk  
import math  

# restore model from airline.nb
model = pickle.load(open('/home/hrrathod/project/naive_bayes/airline.nb', 'rb')) 

# Get list of files in the test set
test_files = sorted(glob.glob('/home/hrrathod/project/test/*.txt'))


# Number of tokens in the positive reviews
pos_total_tokens = model['pos_fd'].N()

# Number of tokens in the neutral reviews
neu_total_tokens = model['neu_fd'].N()

# Number of tokens in the negative reviews
neg_total_tokens = model['neg_fd'].N()


# Combining all FDs
fd = model['pos_fd'] + model['neu_fd'] + model['neg_fd']

# Total number of types in all reviews
vocab_size = model['pos_fd'].B() + model['neu_fd'].B() + model['neg_fd'].B()



# Priors (log)
positive_prior = math.log((model['pos_n']) / (model['pos_n'] + model['neu_n'] + model['neg_n']))
neutral_prior = math.log((model['neu_n']) / (model['pos_n'] + model['neu_n'] + model['neg_n']))
negative_prior = math.log((model['neg_n']) / (model['pos_n'] + model['neu_n'] + model['neg_n']))


# Open the output file
output_file = open('/home/hrrathod/project/naive_bayes/baseline_predictions.txt', 'w')

# Go through all the files in the test set
for file_name in test_files:

    # Open the current file
    current_file = open(file_name, 'r')

    # Read the file 
    text = current_file.read()

    # Tokenize the text in the current file
    tokens = nltk.word_tokenize(text)

    # Initialize the scores for the current file
    # (our total for now is just the prior)
    positive_total = positive_prior
    neutral_total = neutral_prior
    negative_total = negative_prior

    # Now go through each token in the current file,
    # adding the word log probabilities to our totals 
    for t in tokens:

        # Only use words in one of the frequency distributions
        # We already have a combined frequency distribution, so we
        # can just use that.
        if t in fd:
            # P(token|positive), with add-one smoothing
            p_t_positive = (model['pos_fd'][t] + 1) / (pos_total_tokens + vocab_size)

            # P(token|neutral), with add-one smoothing
            p_t_neutral = (model['neu_fd'][t] + 1) / (neu_total_tokens + vocab_size)

            # P(token|negative), with add-one smoothing
            p_t_negative = (model['neg_fd'][t] + 1) / (neg_total_tokens + vocab_size)

            positive_total = positive_total + math.log(p_t_positive)
            neutral_total = neutral_total + math.log(p_t_neutral)
            negative_total = negative_total + math.log(p_t_negative)

   
    # comparing probabilities to classify file as pos/neg
    highestProb = max(positive_total, neutral_total, negative_total)
    if highestProb == positive_total:
        print(current_file.name, '\tpos', file = output_file) # printing results 
    elif highestProb == neutral_total:
        print(current_file.name, '\tneu', file = output_file) # printing results 
    else:
        print(current_file.name, '\tneg', file = output_file) # printing results     
    
                    

# closing the output file
output_file.close()
