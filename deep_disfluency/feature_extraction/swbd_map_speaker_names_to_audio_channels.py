
# coding: utf-8

# In[1]:

#After conversion and audio features have been extracted, rename the split channels according to the intensity values from OpenSmile or IBM ASR results (this is not consistently A: l, B: r in SWBD)


# In[6]:

import os
import sys
import numpy as np
from collections import defaultdict


# In[17]:

#rename all separate channel audio files and feature files based on the acoustic features (intensity?) from the csv files and the word timings
#the one with the highest volume wins


# In[18]:

def get_first_n_timings(timings_file,n=4):
    f = open(timings_file)
    c = 0
    timings = []
    for line in f:
        s = line.split("\t")
        start = float(s[0]); stop = float(s[1])
        timings.append((start,stop))
        c+=1
        if c == n: break   
    f.close()
    return timings


# In[19]:

def get_mean_intensity(filename,intervals=None):
    "frameIndex; frameTime; pcm_RMSenergy_sma; pcm_LOGenergy_sma; F0final_sma;    voicingFinalUnclipped_sma; F0raw_sma; pcm_intensity_sma; pcm_loudness_sma"
    #print intervals
    f = open(filename)
    intensities = []
    start = False
    for line in f:
        #print line
        if start == False:
            start = True; continue
        s = line.split(";")
        #print s
        time = float(s[1])
        #print time
        if intervals[-1][1] < time:
            break
        for i in intervals:
            if time > i[0] and time <= i[1]:
                intensities.append(float(s[8])) #adding rmrs
                break
    f.close()    
    return np.average(intensities) 


# In[24]:

rootdir = "../../../swbd"
audio_feature_files = os.listdir(rootdir+"/audio_features/")
wordtimingdir = rootdir + "/mapping_MS2SWDA/"
wordtimingfiles = os.listdir(wordtimingdir)

missed = []
pair = []

#NB this can be adjusted to rename the separate channel wav files too
for audio in sorted(audio_feature_files):
    dialogue_number = audio[3:7]
    print(dialogue_number)
    if "A.csv" in audio or "B.csv" in audio: continue
    #if not int(dialogue_number) in [2241, 3011]:
    #    continue

    csv = rootdir + "/audio_features/" + audio
    #wavfile = rootdir + "/wav/"+ audio.replace(".csv",".wav") #can add wavs too
    pair.append((csv,None))
    if len(pair)<2: 
        continue
    assert pair[0][0].replace("_l","").replace("_r","") == pair[1][0].replace("_l","").replace("_r",""),\
           pair[0][0].replace("_l","").replace("_r","")+    " " + pair[1][0].replace("_l","").replace("_r","")
    #print "pair", pair
    #we have the pair
    #grab the mapping/timing files for A and B and check the first 3-4 words and get average intensity
    #from each l/r file
    #map each A and B accordingly to which file has the highest intensity for both first 5 words/
    #sanity check is that one should win one, one should win the other
    timing_files = [x for x in wordtimingfiles if dialogue_number in x]
    #print "timing", timing_files
    if not len(timing_files)==2:
        print(dialogue_number, "missing")
        missed.append(csv)
        pair = []
        continue
    intensity_scores = defaultdict(list) #dict of ((wav/opensmilcsv)) -> [intensity at A's words, intensity at B's words]
    for t in timing_files:
        timings = get_first_n_timings(wordtimingdir+t,n=150)
        speaker = t[4:5]
        #print speaker
        for p in pair:
            #print "p", p
            score = get_mean_intensity(p[0],intervals=timings)
            intensity_scores[p[0]].append((speaker,score))
    #print intensity_scores.items()
    winner = {}
    for p in list(intensity_scores.keys()):
        winner[p] = max(intensity_scores[p], key=lambda x: x[1])[0]
    #print winner
    if winner[pair[0][0]] == winner[pair[1][0]]:
        missed.append(csv)
        pair = []
        continue
    #do the renaming
    for p in pair:
        leftright = p[0][-6:-4]
        speaker = winner[p[0]]
        for filename in p:
            if not filename: continue
            c = "mv {} {}".format(filename,filename.replace(leftright,speaker))
            print(c)
            os.system(c)
    pair= [] #reset
    #h = raw_input()
    #if h == "q": break


# In[25]:
print("missed")
print(missed)