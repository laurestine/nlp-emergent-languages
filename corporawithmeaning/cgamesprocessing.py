# This Python program will turn the raw data from the Columbia Games Corpus
# into a .csv file consisting of sentences, meaning representations, and groups (by session of data collection)
# The Columbia Games Corpus is available here: https://catalog.ldc.upenn.edu/LDC2021S02
# This Python program assumes that it is located in the same folder
# as the folder columbia-games-corpus containing the entire corpus as it is downloaded.
import csv
import os

# Ensure the working directory is set appropriately
os.chdir("C:\\Users\\Lola\\Downloads")

# The path to the "data" subfolder of the corpus
datapath = "columbia-games-corpus\\data"

# This list will store (as tuples) the rows of our output talbe.
sentencenmeaning = []

# This is a helper function to decide if two intervals overlap.
def overlapyn(start1,end1,start2,end2):
    return not((end1 <= start2) or (end2 <= start1))

# Loop over all Cards games in all sessions.
for session in ["01","02","03","04","05","06","07","08","09","10","11","12"]:
    sessionpath = datapath+"\\session_"+session

    for game in ["cards.1","cards.2","cards.3"]:
        tasksfilename = "s"+session+"."+game+".tasks"
        wordsAfilename = "s"+session+"."+game+".A.words"
        wordsBfilename = "s"+session+"."+game+".B.words"
        turnsAfilename = "s"+session+"."+game+".A.turns"
        turnsBfilename = "s"+session+"."+game+".B.turns"

        tasklist = []
        with open(sessionpath+"\\"+tasksfilename) as f:
            reader = csv.reader(f,delimiter  = ' ')
            for row in reader:
                tasklist.append(row)

        words = {'A':[],'B':[]}
        with open(sessionpath+"\\"+wordsAfilename) as f:
            reader = csv.reader(f,delimiter=' ')
            for row in reader:
                words['A'].append(row)
        with open(sessionpath+"\\"+wordsBfilename) as f:
            reader = csv.reader(f,delimiter=' ')
            for row in reader:
                words['B'].append(row)
        turns = {'A':[],'B':[]}
        with open(sessionpath+"\\"+turnsAfilename) as f:
            reader = csv.reader(f,delimiter=' ')
            for row in reader:
                turns['A'].append(row)
        with open(sessionpath+"\\"+turnsBfilename) as f:
            reader = csv.reader(f,delimiter=' ')
            for row in reader:
                turns['B'].append(row)

        # We will extract one sentence per Task.
        # We will attempt to select the sentence in which
        # one player is describing the card.
        # Then we will use the card description as the meaning representation.
        for mytask in tasklist:
            # Start and end times for this task
            tstart = float(mytask[0])
            tend = float(mytask[1])

            taskdata = mytask[2].split(';')

            # We filter by only tasks that are part of the game.
            if ('Phase1' in taskdata) or ('Phase2' in taskdata):
                # This will get the list of items on the card.
                taskmeaning = [datum[5:].split(',') for datum in taskdata if 'Card:' in datum][0]

                # This will get 'A' or 'B' depending on who will be describing the card.
                taskspeaker = [datum[10:] for datum in taskdata if 'Describer:' in datum][0]

                # All the non-silent turns overlapping this task.
                taskturns = [turn for turn in turns[taskspeaker]
                                if (overlapyn(tstart,tend,float(turn[0]),float(turn[1])) and turn[2] != '#')]

                # Let's add grouping symbols grouping these turns:
                # Turns of type 'O', 'I', and 'BI' are treated part of their preceding turn.
                group = 0
                for turn in taskturns:
                    if(turn[2] in ['O','I','BI']):
                        turn.append(group)
                    else:
                        group = group + 1
                        turn.append(group)

                # We'll treat the longest turn as being likely the most content-ful,
                # and also include all turns in the same group.
                myturn = max(taskturns, key=lambda l:float(l[1])-float(l[0]))
                groupturns = [turn for turn in taskturns if turn[3]==myturn[3]]
                # Then we'll let all turns in the group determine start/end of what words are relevant.
                turnstart = min([float(turn[0]) for turn in groupturns])
                turnend = max([float(turn[1]) for turn in groupturns])

                # All the non-silent words overlapping this turn.
                turnwords = [word for word in words[taskspeaker]
                                if (overlapyn(turnstart,turnend,float(word[0]),float(word[1])) and word[2] != '#')]

                # Then we'll just join all the selected words together!
                tasksentence = ' '.join([word[2] for word in turnwords])

                sentencenmeaning.append((tasksentence,taskmeaning, session))

with open('cgames-combined.csv','w') as f:
    f.write("mr,ref,group\n" +
            "\n".join([';'.join(line[1])+","+line[0]+","+line[2] for line in sentencenmeaning]))