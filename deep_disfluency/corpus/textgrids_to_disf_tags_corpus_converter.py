import os
import sys
from collections import Counter, defaultdict
from mumodo.mumodoIO import open_intervalframe_from_textgrid
from copy import deepcopy

"""
Methods to consume textgrids and convert to the disfluency
corpus style for consistency across different possible raw formats.

This file is distributed as part of DUEL corpus.
"""

def disfluency_tags(utt, corpus, start_time):
    """returns the list of tags for each word (simply defined by split)
    and also the list of tags for boundaries (one more than the utt length) 
    for repair points and laughter bouts. NB problem is: the laughter bout itself is a word
    may in fact instead need to do this after we establish which words are proper words"""
    utt = utt.split()
    labels = ["",] * len(utt)
    boundaries = ["",] * (len(utt)+1)
    inReparandum = 0
    inRepair = 0
    inFP = True
    inLS = False
    for i in range(0,len(utt)):
        word = utt[i]
        #if word == "-":
        
        if "<laughter>" in word or "<laugther>" in word:
            inLS = True
        if "<p" in word:
            labels[i]+="<p/>"
        for j in range(0,len(word)):
            filled_pause_begin = False
            c = word[j]
            if c=="(":
                inReparandum+=1
            if c == "{":
                #if j == len(word)-1:
                #    pass #edit term (non-fp)
                #elif word[j+1] == "F":
                #    inFP=True
                #else:
                inFP = True
                filled_pause_begin = True
        if inFP or filled_pause_begin:
            labels[i]+="<e/>"
        if inReparandum>0 and inFP==False:
            labels[i]+="<rm/>"
        if inRepair>0 and inFP==False:
            labels[i]+="<rp/>"
        if inLS==True:
            labels[i]+="<ls/>"
        if "</laughter>" in word:
            inLS=False
            
        for j in range(0,len(word)):
            c = word[j]
            if c == ")": inRepair-=1 # for now counting interegnum within the repairs
            if c == "+": inRepair+=1; inReparandum-=1
            if c =="}": #out of the filled pause
                inFP=False
                       
    #if inLS == True:
    #    print "WARNING NO LS END", corpus, start_time
        #raw_input()
    return labels

def convert_to_disfluency_word_tag_tuples_from_raw(text, start='(', end=')'):
    """ Turn the raw transcript with Shriberg-style bracketed format to
    the disfluency detection format. i.e. from:

    "John ( likes, + {F uh, } loves ) Mary"

    to:

    [('John',  '<f/>'),
     ('likes', '<rms id="3"/>'),
     ('uh',    '<e/><i id="3"/>'),
     ('loves', '<rps id="3"/>'),
    `('Mary',  '<f/>')
    ]
    """
    depth = 0  # if not 0, then too many/too few brackets
    word_tag_tuples = []
    inReparandum = 0
    inRepair = 0
    inFP = True
    inLS = False
    for i, word in enumerate(text.split()):
        word = clean_word(word)
        if word == "":
            continue
        if "<laughter>" in word or "<laugther>" in word:
            inLS = True
        label = ""
        for j in range(0, len(word)):
            filled_pause_begin = False
            c = word[j]
            if c == start:
                inReparandum += 1
            if c == "{":
                inFP = True
                filled_pause_begin = True
        if inFP or filled_pause_begin:
            label += "<e/>"
        if inReparandum > 0 and inFP is False:
            label += "<rm/>"
        if inRepair > 0 and inFP is False:
            label += "<rp/>"
        if inLS is True:
            label += "<ls/>"
        if "</laughter>" in word:
            inLS = False

        for j in range(0, len(word)):
            c = word[j]
            if c == end:
                inRepair -= 1  # for now counting interegnum within the repairs
            if c == "+":
                inRepair += 1
                inReparandum -= 1
            if c == "}":  # out of the filled pause
                inFP = False
        if label == "":
            label = "<f/>"
    if inRepair != 0:
        print("non 0 bracket depth/too few or too many!", inRepair)
    return inRepair



lang = "de"
duel_dir =  "../../../../../duel_transcriptions/"
transcription_dir = duel_dir + "{}/transcriptions_annotations/".format(lang)
target_dir = "../../../../../DUEL/{}".format(lang)

task_index = {
    1 : "dream_apartment",
              2: "film_script",
              3: "border_control"
             }

legal_tiers = {"A-utts" : [u"A", u"A-utts;"], 
               "B-utts" : [u"B", u"B-utts;", u"B_utts"], 
               "A-turns" : [u"A-turns;","A_turns"], 
               "B-turns" : [ u"B-turns;",u"B_turns", u"B-turns    "],
               "A-laughter" : [], 
               "B-laughter" : [u"B−laughter"],
               "A-en" : [u"A-eng", u"A-english",
                         u"A-fr_en", u"A-fr-en",
                         u"A-fr_en;",u"Translation A",
                         u"translation A", u"A translation", u"A Translation"], 
               "B-en" : [u"B-eng", u"B-english",
                         u"B-fr_en", u"B-fr_en;",
                         u"B_fr-en", u"Translation B", 
                         u"translation B", u"B translation",
                         u"B Translation", u"B-fr-en"],
               "Comments" : [u"Comments & questions",
                             u"comments", u"Problems"], 
               "Part" : [u"part"], 
               "O" : [u"E"]
              }

c = Counter()
missing_c = defaultdict(list)
global_tag_count = Counter()
log_file = open("{}_errors.log".format(lang), "w")

for experiment_name in sorted(os.listdir(transcription_dir)):
    print(experiment_name)
    if ".DS_Store" in experiment_name:
        continue
    textgrid_file_name = transcription_dir + os.sep + experiment_name + os.sep + experiment_name + ".TextGrid"
    textgrid_file_name_target = target_dir + os.sep + experiment_name + os.sep + experiment_name + ".TextGrid"
    print(textgrid_file_name)
    tasks = task_index.values() #NB idio syncratic to DUEL
    if not os.path.isfile(textgrid_file_name):
        #already sliced, so just add all the corresponding files
        if not os.path.isdir(target_dir+os.sep+experiment_name):
            os.mkdir(target_dir+os.sep+experiment_name)
        for textgrid in sorted(os.listdir(transcription_dir + os.sep + experiment_name)):
            textgrid_file_name = transcription_dir + os.sep + experiment_name + os.sep + textgrid
            task_number = textgrid_file_name[textgrid_file_name.rfind("_")+1:
                                             textgrid_file_name.rfind(".")]
            # if not ".TextGrid" in textgrid_file_name or
            # ("TODO" in textgrid_file_name or "full" in textgrid_file_name): 
            if not "correctedTODO.TextGrid" in textgrid_file_name:
                #print "ommitting", textgrid_file_name
                continue
            print("File:",textgrid_file_name)
            #part_transcription = open_intervalframe_from_textgrid(textgrid_file_name)
            #for k in part_transcription.keys():
            #    c[k]+=1
            #try:
            if True:
                textgrid_file_name_target = target_dir + os.sep + experiment_name + os.sep + textgrid.replace(
                    "_correctedTODO", "")
                #part_transcription = open_intervalframe_from_textgrid(textgrid_file_name)
                trans, missing = verify_file(textgrid_file_name, global_tag_count, log_file, translator)
                #WARNING WARNING be careful to not overwrite!!
                print("saving to", textgrid_file_name_target)
                print("names:", trans.keys())
                
                # save_intervalframe_to_textgrid(trans,textgrid_file_name_target)
                
                #save_intervalframe_to_textgrid(trans,textgrid_file_name./
                # replace(".TextGrid","_correctedTODO.TextGrid"))
                missing_c[textgrid].extend(missing)
            #except Exception:
            #    print "couldn't verify", textgrid_file_name
                #continue

    else:
        if not os.path.isdir(target_dir+os.sep+experiment_name):
            os.mkdir(target_dir+os.sep+experiment_name)
        #quirk of German
        textgrid_file_name = textgrid_file_name.replace(".TextGrid",
                                                        "_correctedTODO.TextGrid")
        if not "correctedTODO.TextGrid" in textgrid_file_name:
            #print "ommitting", textgrid_file_name
            continue
        print("File",textgrid_file_name)
        trans, missing = verify_file(textgrid_file_name, global_tag_count, log_file, translator)
        missing_c[experiment_name].extend(missing)
        #WARNING WARNING be careful to not overwrite!!
        #save_intervalframe_to_textgrid(trans,textgrid_file_name)
        print("saving to", textgrid_file_name_target)
        print("names", trans.keys())
        
        # save_intervalframe_to_textgrid(trans, textgrid_file_name_target)


        #just one file, so slice into tasks/parts according to the 'part tier'
        #transcription = open_intervalframe_from_textgrid(textgrid_file_name)
print("*********tiers")
for k,v in sorted(c.items(),key=lambda x: x[1], reverse=True):
    if v == []: continue
    print(k,v)
print("******missing")
for k,v in sorted(missing_c.items(),key=lambda x: x[0]):
    if v == []: continue
    print(k,v)
print("*********tags")
for k,v in sorted(global_tag_count.items(),key=lambda x: x[1], reverse=True):
    if v == []: continue
    print(k,v)
log_file.close()


if __name__ == '__main__':
    # 
    pass
