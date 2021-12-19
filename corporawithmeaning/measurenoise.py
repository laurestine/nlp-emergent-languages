# This Python program will turn the raw data from the Columbia Games Corpus
# into a .csv file consisting of the voice intensity and noise intensity, for each speaker in each session.
# The Columbia Games Corpus is available here: https://catalog.ldc.upenn.edu/LDC2021S02
# This Python program assumes that it is located in the same folder
# as the folder columbia-games-corpus containing the entire corpus as it is downloaded.
import soundfile as sf # Processing .flac audio files
import numpy as np
import csv
import math # need ceiling function
#import os

# Ensure the working directory is set appropriately
# os.chdir("C:\\Users\\Lola\\Downloads")

# The path to the "data" subfolder of the corpus
datapath = "columbia-games-corpus\\data"

# We will need to map to "the other person's" audio to get the right files.
opposite = {'A':'B','B':'A'}

allsignoises = []
sessionsignoises = []

# One session at a time first.
for session in ["01","02","03","04","05","06","07","08","09","10","11","12"]:

    sessionsignal = {'A':[],'B':[]}
    sessionnoise = {'A':[],'B':[]}

    for game in ["cards.1","cards.2","cards.3"]:

        # Read in all the data!!

        sound = {'A':sf.read(datapath+"\\session_"+session+"\\s"+session+"."+game+".A.flac"), # a tuple, data first, sample rate 2nd
                'B':sf.read(datapath+"\\session_"+session+"\\s"+session+"."+game+".B.flac")}

        turns = {'A':[],'B':[]}
        with open(datapath+"\\session_"+session+"\\s"+session+"."+game+".A.turns") as f:
            reader = csv.reader(f,delimiter=' ')
            for row in reader:
                turns['A'].append(row)
        with open(datapath+"\\session_"+session+"\\s"+session+"."+game+".B.turns") as f:
            reader = csv.reader(f,delimiter=' ')
            for row in reader:
                turns['B'].append(row)

        # Now we need to highlight intervals when A is talking, intervals when B is talking,
        # and intervals when neither is talking.
        # We'll get the intervals first,
        # then fill in the timepoints later.

        # bigset is a set of intervals (tuples of endpoints) in increasing order
        # interval is an interval (tuple of endpoints)
        # we return what you get by taking the union of bigset with interval.
        def intervalunionwithset(bigset, interval):
            newstart = interval[0]
            newend = interval[1]
            for int2 in bigset:
                if int2[0] <= interval[0] and int2[1] >= interval[0]:
                    newstart = int2[0]
                if int2[0] <= interval[1] and int2[1] >= interval[1]:
                    newend = int2[1]
            newinterval = (newstart,newend)
            toggle = 0
            answer = []
            for int2 in bigset:
                if int2[1] < newinterval[0]:
                    answer.append(int2)
                if int2[0] > newinterval[1]:
                    if not toggle:
                        answer.append(newinterval)
                        toggle=1
                    answer.append(int2)
            if not toggle:
                answer.append(newinterval)
            return answer

        # bigset is a set of intervals in increasing order,
        # intervals is a set of intervals,
        # we get a set of intervals which is the union, in increasing order.
        def intervalsunionwithset(bigset,intervals):
            answer = bigset
            for interval in intervals:
                answer = intervalunionwithset(answer,interval)
            return(answer)

        # universe is an interval (a tuple of endpoints),
        # intervals a set of intervals in increasing order,
        # we get a set of intervals which is the complement of intervals in universe.
        def intervalscomplement(intervals,universe):
            answer = []
            start = universe[0]
            for interval in intervals:
                end = interval[0]
                answer.append((start,end))
                start = interval[1]
            end = universe[1]
            answer.append((start,end))
            return(answer)

        speechintervals = {}
        for speaker in ['A','B']:
            speechintervals[speaker] = [(float(turn[0]),float(turn[1]))
                                            for turn in turns[speaker] if turn[2] != '#']

        # Noise intervals are when neither player is talking.
        noiseintervals = intervalscomplement(
            intervalsunionwithset(speechintervals['A'],speechintervals['B']),
            (0,len(sound['A'][0])/sound['A'][1]))

        # This treats it as a half-open interval [endpoint0, endpoint1), and gives
        # the indeces of the relevant samples, when sampled at the given frequency
        def intervaltoindeces(endpoints,frequency):
            return range(math.ceil(endpoints[0]*frequency),math.ceil(endpoints[1]*frequency))

        # Now we will get the indeces of timepoints where each person is talking (speechindeces),
        # or when no one is talking (noiseindeces)
        speechindeces = {}
        for speaker in ['A','B']:
            speechindeces[speaker] = []
            for interval in speechintervals[speaker]:
                speechindeces[speaker] = speechindeces[speaker] + list(
                    intervaltoindeces(interval, sound[opposite[speaker]][1]))
        noiseindeces = []
        for interval in noiseintervals:
            noiseindeces = noiseindeces + list(
                    intervaltoindeces(interval, sound['A'][1]))

        # Then we will record the average intensity
        # in B's mic when A is speaking vs when no one is speaking,
        # and in A's mic when B is speaking vs when no one is speaking.
        intnnoise = {}
        for speaker in ['A','B']:
            sessionsignal[speaker] = sessionsignal[speaker] + [abs(sound[opposite[speaker]][0][i]) for i in speechindeces[speaker]]
            sessionnoise[speaker] = sessionnoise[speaker] + [abs(sound[opposite[speaker]][0][i]) for i in noiseindeces]
            intnnoise[speaker] = (np.mean([abs(sound[opposite[speaker]][0][i]) for i in speechindeces[speaker]]),
                                    np.mean([abs(sound[opposite[speaker]][0][i]) for i in noiseindeces]))
            allsignoises.append((session, game, speaker, intnnoise[speaker][0],intnnoise[speaker][1]))
    # We recorded average by game, but
    # we're mainly going to use the average by session.
    for speaker in ['A','B']:
        sessionsignoises.append((session, speaker,np.mean(sessionsignal[speaker]),np.mean(sessionnoise[speaker])))

#with open("signalnoise.csv",mode="w") as f:
#    f.write("session,game,speaker,voice,noise\n"+"\n".join(
#        [datum[0]+","+datum[1]+","+datum[2]+","+str(datum[3])+","+str(datum[4]) for datum in allsignoises]
#    ))

with open("sessionsignalnoise.csv",mode="w") as f:
    f.write("session,speaker,voice,noise\n"+"\n".join(
        [datum[0]+","+datum[1]+","+str(datum[2])+","+str(datum[3]) for datum in sessionsignoises]
    ))

#for datum in sessionsignoises:
#    print(datum[0],datum[1],str(datum[2]/datum[3]))