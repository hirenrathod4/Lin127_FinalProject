import pickle
import glob
import nltk  
from nltk.tokenize import MWETokenizer


# Restore list of collocations from collocation.nb
model = pickle.load(open('/home/hrrathod/project/collocations/collocation.nb', 'rb')) 

# REPLACE PATH
# Getting a list of file names (with their paths) in a specific directory
pos_files = glob.glob('/home/hrrathod/project/train/pos/*.txt')
neu_files = glob.glob('/home/hrrathod/project/train/neu/*.txt')
neg_files = glob.glob('/home/hrrathod/project/train/neg/*.txt')



# Creat empty list for the multi-word expression tokens
tokenizer = MWETokenizer([])
# Converting the collocations into a multi-word expression token
for w1, w2 in model:
    tokenizer.add_mwe((w1, w2))


all_ptokens = []

# Looping through each file path in the pos directory
for pos_review in pos_files:
  
  opening_pos = open(pos_review, 'r')
  reading_pos = str(opening_pos.read())
  
  # tokenizes every positive review (lists of tokens)
  p_tokens = tokenizer.tokenize(reading_pos.split())
  
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
  reading_neu = str(opening_neu.read())
  
  # tokenizes every neutral review (lists of tokens)
  neu_tokens = tokenizer.tokenize(reading_neu.split())
  
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
  reading_neg = str(opening_neg.read())
  
  # tokenizes every negative review (lists of tokens)
  n_tokens = tokenizer.tokenize(reading_neg.split())
  all_ntokens = all_ntokens + n_tokens
  
# frequency distribution of the negative files
neg_fd = nltk.FreqDist(all_ntokens)

# the number of negative samples
number_of_neg = len(neg_files)



# a dictionary that consists of the frequency distribution
# for each class and the number of samples in each class
model = {'pos_n': number_of_pos,
         'neu_n': number_of_neu,
         'neg_n': number_of_neg,
         'pos_fd': pos_fd,
         'neu_fd': neu_fd,
         'neg_fd': neg_fd}

# saving model
pickle.dump(model, open('/home/hrrathod/project/collocations/train_collocation.nb', 'wb')) 




