def main():
    
    key = open('/home/hrrathod/project/testKey.txt', 'r')
    # Always replace to any predictions file that you are comparing to the test key
    mine = open('/home/hrrathod/project/naive_bayes/baseline_predictions.txt', 'r')
    # Files where we are only writing the label to from the key and the prediction file
    keyOut = open('/home/hrrathod/project/key.txt', 'w')
    mineOut = open('/home/hrrathod/project/mine.txt', 'w')
    predName = mine.name
    keyList = []
    mineList = []
    totalFiles = 0
    #Iterating through every line in the test key 
    for line in key:
        totalFiles = totalFiles + 1
        for word in line.split():
            # Only outputting the label to an output file
            if word == 'pos' or word == 'neu' or word == 'neg':
                print(word, file=keyOut)
                keyList += word
    #Iterating through every line in the predictions file 
    for line in mine:
        for word in line.split():
            # Only outputting the label to an output file
            if word == 'pos' or word == 'neu' or word == 'neg':
                print(word, file=mineOut)
                mineList += word

    keyOut.close()
    mineOut.close()

    # Open the files that we just wrote to that contain only the labels
    key = open('/home/hrrathod/project/key.txt', 'r')
    mine = open('/home/hrrathod/project/mine.txt', 'r')
    # Will accumulate incorrect classifications
    mismatch = 0
    for line1, line2 in zip(key, mine):
        # Comparing line by line, if they dont match, then the classification is incorrect
        if line1 != line2:
            mismatch += 1
    # Computing the simple accuracy
    accuracy = ((totalFiles - mismatch) / totalFiles) * 100
    # Setting the precision
    accuracy = round(accuracy, 3)
    # Outputting Accuracy Results
    print("Computing accuracy for:", predName)
    print("Total Files: ", totalFiles) 
    print("Incorrect Classifications: ", mismatch)
    print("Accuracy: ", str(accuracy) + "%")


   

if __name__ == "__main__":  
    main()