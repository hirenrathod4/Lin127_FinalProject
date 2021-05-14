import csv
import os

##### READ ME:

# Start by creating a directory called "project" so that your current directory is "/home/<your username>/project/"
# Add this python file to the directory above
# Add Tweets.csv to your directory
# Use: mkdir train, mkdir dev, and mkdir test in your terminal to make the directories for train, dev, and test
# In train, use: mkdir pos, mkdir neu, and mkdir test to create those training sets
# Replace my paths with yours where I've commented throughout the code

#####

def main():   
    #Replace path below
    csvFile = open('/home/hrrathod/project/Tweets.csv', 'r')
    csvRead = csv.reader(csvFile, delimiter = ',')
    i = 0
    string = "0"
    
    #Replace path below
    outFile = "/home/hrrathod/project/"
    #Replace path below
    outFileTrain = open('/home/hrrathod/project/trainKey.txt', 'w')
    #Replace path below
    outFileDev = open('/home/hrrathod/project/devKey.txt', 'w')
    #Replace path below
    outFileTest = open('/home/hrrathod/project/testKey.txt', 'w')
    #Replace path below
    os.chdir("/home/hrrathod/project/train")
    for line in csvRead:
        string = string + str(i)
        if i < 5000:
            if line[1] == "positive":
                #Replace path below
                os.chdir("/home/hrrathod/project/train/pos/")
                #Replace path below
                outPath = '/home/hrrathod/project/train/pos/' + string + '.txt'
                outFile = open(outPath, 'w')
                print(line[10], file=outFile)
                outFile.close()
                print('pos\t', 'train/pos/' + string + '.txt', file=outFileTrain)
                
            elif line[1] == "neutral":
                #Replace path below
                os.chdir("/home/hrrathod/project/train/neu/")
                #Replace path below
                outPath = '/home/hrrathod/project/train/neu/' + string + '.txt'
                outFile = open(outPath, 'w')
                print(line[10], file=outFile)
                outFile.close()
                print('neu\t', 'train/neu/' + string + '.txt', file=outFileTrain)
                
            else:
                #Replace path below
                os.chdir("/home/hrrathod/project/train/neg/")
                #Replace path below
                outPath = '/home/hrrathod/project/train/neg/' + string + '.txt'
                outFile = open(outPath, 'w')
                print(line[10], file=outFile)
                outFile.close()
                print('neg\t', 'train/neg/' + string + '.txt', file=outFileTrain)

            outFile.close()
            if i == 4999:
                outFileTrain.close()
                #Replace path below
                os.chdir("/home/hrrathod/project/dev")
                
        elif i < 10000:
            #Replace path below
            outPath = '/home/hrrathod/project/dev/' + string + '.txt'
            outFile = open(outPath, 'w')
            if line[1] == "positive":
                print('pos\t', 'dev/' + string + '.txt', file=outFileDev)
            elif line[1] == "neutral":
                print('neu\t', 'dev/' + string + '.txt', file=outFileDev)
            else:
                print('neg\t', 'dev/' + string + '.txt', file=outFileDev)
            print(line[10], file=outFile)
            outFile.close()
            if i == 9999:
                outFileDev.close()
                #Replace path below
                os.chdir("/home/hrrathod/project/test")    
        else:
            #Replace path below
            outPath = '/home/hrrathod/project/test/' + string + '.txt'
            outFile = open(outPath, 'w')
            if line[1] == "positive":
                print('pos\t', 'test/' + string + '.txt', file=outFileTest)
            elif line[1] == "neutral":
                print('neu\t', 'test/' + string + '.txt', file=outFileTest)
            else:
                print('neg\t', 'test/' + string + '.txt', file=outFileTest)
            print(line[10], file=outFile)
            outFile.close()
        string = "0"       
        i += 1
    outFileTest.close()
    
    

if __name__ == "__main__":    
    main()