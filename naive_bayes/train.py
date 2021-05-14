import glob
import nltk                   
import pickle

# Getting a list of file names (with their paths) in a specific directory
pos_files = glob.glob('/home/hrrathod/project/train/pos/*.txt')
neu_files = glob.glob('/home/hrrathod/project/train/neu/*.txt')
neg_files = glob.glob('/home/hrrathod/project/train/neg/*.txt')

# Mapping the number of files that the keys appear in within their respective sets (pos, neu, neg)
posTypeCounts = {}
neuTypeCounts = {}
negTypeCounts = {}

# Map storing all of the positive tokens
all_ptokens = []

# Looping through each file path in the pos directory
for pos_review in pos_files:
  opening_pos = open(pos_review, 'r')
  reading_pos = opening_pos.read()
  # tokenizes every positive review (lists of tokens)
  p_tokens = nltk.word_tokenize(reading_pos)
  # Appends tokens the we have already looked at from the tokenization above
  strMap = {}
  # Iterating through all of the tokens
  for token in p_tokens:
      # Checks to see if we have already examined a potentially recurring token
      value1 = strMap.get(token)
      # If it is not a token that we have seen
      if value1 == None:
        # Checking to see if this type exists in our pos type map 
        value2 = posTypeCounts.get(token)
        # The token does not exist in the map abov
        if value2 == None:
            # Adding an instance of the token to the tfidf map
            posTypeCounts[token] = 1
        else:
            # Incrementing the value in the map to reflect that the current token is present in the current file
            posTypeCounts[token] = posTypeCounts[token] + 1
        # Creating an entry for the token that we have just observed
        strMap[token] = True
  # getting a list of all the positive tokens
  all_ptokens = all_ptokens + p_tokens


# frequency distribution of the positive files
pos_fd = nltk.FreqDist(all_ptokens)
# the number of positive samples
number_of_pos = len(pos_files)

all_neutokens = []
# Looping through each file path in the neu directory
for neu_review in neu_files:
  opening_neu = open(neu_review, 'r')
  reading_neu = opening_neu.read()
  # tokenizes every neutral review (lists of tokens)
  neu_tokens = nltk.word_tokenize(reading_neu)
  # Appends tokens the we have already looked at from the tokenization above
  strMap = {}
  # Iterating through all of the tokens
  for token in neu_tokens:
      # Checks to see if we have already examined a potentially recurring token
      value1 = strMap.get(token)
      # If it is not a token that we have seen
      if value1 == None:
        # Checking to see if this type exists in our pos type map 
        value2 = neuTypeCounts.get(token)
        # The token does not exist in the map abov
        if value2 == None:
            # Adding an instance of the token to the tfidf map
            neuTypeCounts[token] = 1
        else:
            # Incrementing the value in the map to reflect that the current token is present in the current file
            neuTypeCounts[token] = neuTypeCounts[token] + 1
        # Creating an entry for the token that we have just observed
        strMap[token] = True
  # getting a list of all the neutral tokens
  all_neutokens = all_neutokens + neu_tokens

# frequency distribution of the neutral files
neu_fd = nltk.FreqDist(all_neutokens)
# the number of neutral samples
number_of_neu = len(neu_files)

all_ntokens = []
# Looping through each file path in the neg directory
for neg_review in neg_files:
  opening_neg = open(neg_review, 'r')
  #reading_neg = str(opening_neg.read())
  reading_neg = opening_neg.read()
  # tokenizes every negative review (lists of tokens)
  n_tokens = nltk.word_tokenize(reading_neg)
  # Appends tokens the we have already looked at from the tokenization above
  strMap = {}
  # Iterating through all of the tokens
  for token in n_tokens:
      # Checks to see if we have already examined a potentially recurring token
      value1 = strMap.get(token)
      # If it is not a token that we have seen
      if value1 == None:
        # Checking to see if this type exists in our pos type map 
        value2 = negTypeCounts.get(token)
        # The token does not exist in the map abov
        if value2 == None:
            # Adding an instance of the token to the tfidf map
            negTypeCounts[token] = 1
        else:
            # Incrementing the value in the map to reflect that the current token is present in the current file
            negTypeCounts[token] = negTypeCounts[token] + 1
        # Creating an entry for the token that we have just observed
        strMap[token] = True
  all_ntokens = all_ntokens + n_tokens

# frequency distribution of the negative files
neg_fd = nltk.FreqDist(all_ntokens)
# the number of negative samples
number_of_neg = len(neg_files)

# a dictionary that consists of the frequency distribution
# for each class and the number of samples in each class
model = {
    'pos_n': number_of_pos,
    'neu_n': number_of_neu,
    'neg_n': number_of_neg,
    'pos_fd': pos_fd,
    'neu_fd': neu_fd,
    'neg_fd': neg_fd, 
    'pos_docs': posTypeCounts, 
    'neu_docs': neuTypeCounts, 
    'neg_docs': negTypeCounts}

# saving model
pickle.dump(model, open('/home/hrrathod/project/naive_bayes/airline.nb', 'wb')) 
