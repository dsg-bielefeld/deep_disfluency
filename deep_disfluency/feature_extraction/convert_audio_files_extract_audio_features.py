
# coding: utf-8

# In[1]:

# Script to convert stereo Switchboard .sph files to two wav files with the correct particpant name, e.g. sw02005A
# Dependencies: SOX and OPENSMILE
# Does this via the following steps:
# 1. Use SOX to convert .sph to .wav
# 2. Use SOX to split channels into 2 separate .wav files (ideally remove cross talk noise too)
# 3. Get audio features from OpenSmile for each channel (TODO could also get IBM ASR results too)


# In[6]:

import os
import sys
import numpy as np
from collections import defaultdict


# In[4]:

# for x in *.wav; do sox $x -e signed-integer wav/$x; done
#rootdir = "/media/dsg-labuser/My Passport/bielefeld-server/External/Switchboard/switchboard_audio/"
rootdir = "/home/dsg-labuser/bielefeld_server/switchboard_audio/"
sep_dir = rootdir + "split_channels/"


# In[2]:

converted_to_wav = True  # turn off and on
if not converted_to_wav:
    sph_files = os.listdir(rootdir)
    sph_files = filter(lambda x: x[-4:] == ".sph", sph_files)
    for sph in sph_files:
        c = 'sox ' + '"' + rootdir + sph + '"' + " -e signed-integer " + '"' + rootdir + "wav/" + sph.replace(".sph", ".wav") + '"'
        print c
        os.system(c)


# In[ ]:

# step 2- split the channel into left and right channels, then remove the corresponding overall wav file above
# to save space
wavfiles = os.listdir(rootdir + "wav/")
start = False
for wav in sorted(wavfiles):
    conv_num = int(wav[-8:-4])
    if conv_num == 2466:
        start = True
    if not start:
        continue
    print wav
    wavfile = rootdir + "wav/" + wav
    commandl = 'sox ' + '"' + wavfile + '" ' + '"' + sep_dir + wav.replace(".wav", "_l.wav") + '" ' + ' remix 1'
    commandr = 'sox ' + '"' + wavfile + '" ' + '"' + sep_dir + wav.replace(".wav", "_r.wav") + '" ' + ' remix 2'
    #commanddel = "rm " + rootdir+"wav/"+wav
    print wavfile
    # print commandl
    os.system(commandl)
    # print commandr
    os.system(commandr)
    # print commanddel
    #h = raw_input()
    # if h == "q": break


# In[9]:

# step 3.1 get the IBM ASR results for each one- could take a long time..
rootdir = sep_dir
wavfiles = os.listdir(rootdir + "/wav")
asr_dir = rootdir + "/asr"
for wav in sorted(wavfiles):
    wavfile = rootdir + "wav/" + wav
    print wavfile


# In[7]:

# step 3.2- get the opensmile features
rootdir = sep_dir
config = "/users/julianhough/git/simple_rnn_disf/rnn_disf_detection/data/combi_prosody.conf"
wavfiles = os.listdir(rootdir + "/wav")
#rootdir = "/Volumes/My\\ Passport/bielefeld-server/External/Switchboard/switchboard_audio/"

for wav in sorted(wavfiles):
    wavfile = rootdir + "wav/" + wav
    csv = wavfile.replace("/wav/", "/audio_features/").replace(".wav", ".csv")
    c = '/Applications/openSMILE-2.1.0/inst/bin/SMILExtract -nologfile -C {} -I "{}" -O "{}"'.format(config, wavfile, csv)
    # print c
    print wavfile
    os.system(c)
    #h = raw_input()
    # if h == "q": break


# In[ ]:
