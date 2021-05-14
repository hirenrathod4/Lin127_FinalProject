import nltk
from nltk.tokenize import MWETokenizer
import pickle

# REPLACE PATH
# Open, read, and tokenize all the text files in the training set
train_file = open('/home/hrrathod/project/collocations/train_set.txt', 'r')
train_text = train_file.read()
train_tokens = nltk.word_tokenize(train_text)

# REPLACE PATH
# Open read, and tokenize all the text files in the test set
test_file = open('/home/hrrathod/project/collocations/test_set.txt', 'r')
test_text = test_file.read()
test_tokens = nltk.word_tokenize(test_text)

# Combine tokens from the training set and the test set
tokens = train_tokens + test_tokens

# Find unigram freq. distribution for all tokens
unigram_fd = nltk.FreqDist(tokens)
# Find bigram freq. distribution for all tokens
bigram_fd = nltk.FreqDist(nltk.bigrams(tokens))

# Create empty list to store all collocations
collocations = []


# Find collacations by looking at the bigram freq. distribution
for w1, w2 in bigram_fd:

    # Added so we look at words that appear more than 100 times
    if unigram_fd[w1] > 100 and unigram_fd[w2] > 100:
        pw1 = unigram_fd[w1] / unigram_fd.N()
        pw2 = unigram_fd[w2] / unigram_fd.N()
        pw1w2 = bigram_fd[w1, w2] / bigram_fd.N()
        # p(w1, w2) / p(w1)p(w2)
        score = (pw1w2 / (pw1 * pw2))
        collocations.append((score, (w1, w2)))

# Sort collocation by score from lowest --> highest
collocations.sort()

# Create empty list to store the collocations we want to use
colloc_list = []

# Adding only the highest scoring 100 collocations to colloc_list 
for score, colloc in collocations[-100:]:
    colloc_list.append(colloc)



model = colloc_list

# saving model
pickle.dump(model, open('/home/hrrathod/project/collocations/collocation.nb', 'wb')) 


