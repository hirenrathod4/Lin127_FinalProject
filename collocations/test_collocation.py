import pickle
import glob
import math  
import nltk  
from nltk.tokenize import MWETokenizer


# restore model from collocation.nb
model_1 = pickle.load(open('/home/hrrathod/project/collocations/collocation.nb', 'rb')) 
# restore model from train_collocation.nb
model_2 = pickle.load(open('/home/hrrathod/project/collocations/train_collocation.nb', 'rb')) 


# Creat empty list for the multi-word expression tokens
tokenizer = MWETokenizer([])
# Converting the collocations into a multi-word expression token
for w1, w2 in model_1:
    tokenizer.add_mwe((w1, w2))


# REPLACE PATH
# Get list of files in the test set
test_files = sorted(glob.glob('/home/hrrathod/project/test/*.txt'))


# Number of tokens in the positive reviews
pos_total_tokens = model_2['pos_fd'].N()
# Number of tokens in the neutral reviews
neu_total_tokens = model_2['neu_fd'].N()
# Number of tokens in the negative reviews
neg_total_tokens = model_2['neg_fd'].N()

# Combining all FDs
fd = model_2['pos_fd'] + model_2['neu_fd'] + model_2['neg_fd']

# Total number of types in all reviews
vocab_size = model_2['pos_fd'].B() + model_2['neu_fd'].B() + model_2['neg_fd'].B()



# Priors (log)
positive_prior = math.log((model_2['pos_n']) / (model_2['pos_n'] + model_2['neu_n'] + model_2['neg_n']))
neutral_prior = math.log((model_2['neu_n']) / (model_2['pos_n'] + model_2['neu_n'] + model_2['neg_n']))
negative_prior = math.log((model_2['neg_n']) / (model_2['pos_n'] + model_2['neu_n'] + model_2['neg_n']))


# Open the output file
output_file = open('/home/hrrathod/project/collocations/collocation_predictions.txt', 'w')

# Go through all the files in the test set
for file_name in test_files:

    # Open the current file
    current_file = open(file_name, 'r')

    # Read the file 
    text = current_file.read()

    # Tokenize the text in the current file
    tokens = tokenizer.tokenize(text.split())
    

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
            p_t_positive = (model_2['pos_fd'][t] + 1) / (pos_total_tokens + vocab_size)

            # P(token|neutral), with add-one smoothing
            p_t_neutral = (model_2['neu_fd'][t] + 1) / (neu_total_tokens + vocab_size)

            # P(token|negative), with add-one smoothing
            p_t_negative = (model_2['neg_fd'][t] + 1) / (neg_total_tokens + vocab_size)

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


