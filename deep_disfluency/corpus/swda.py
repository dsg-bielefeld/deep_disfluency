# -*- coding: utf-8 -*-
"""
Classes for dealing with the CSV version of the Switchboard Dialog Act
corpus that was distributed as part of this course:
http://compprag.christopherpotts.net/.  That CSV version pools the
Dialog Act Corpus, the corresponding Penn Treebank 3 trees, and the
metadata tables from the original Switchboard 2 release.

The main classes are CorpusReader, Transcript, and Utterance.  The
CorpusReader works with the entire corpus, providing iterator methods
for moving through all of the Transcript or Utterance instances in it.
"""

__author__ = "Christopher Potts"
__copyright__ = "Copyright 2011, Christopher Potts"
__credits__ = []
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 \
Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__version__ = "1.0"
__maintainer__ = "Christopher Potts"
__email__ = "See the author's website"

######################################################################

import os
import sys
import csv
import re
from glob import iglob
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer

######################################################################

class Metadata:
    """
    Basically an internal method for organizing the tables of metadata
    from the original Switchboard transcripts and linking them with
    the dialog acts.
    """    
    def __init__(self, metadata_filename):
        """
        Turns the CSV file into a dictionary mapping Switchboard
        conversation_no integers values to dictionaries of values. All
        the keys correspond to the column names in the original
        tables.

        Argument:

        metadata_filename (str) -- the CSV file swda-metadata.csv
        (should be in the main folder of the swda directory)
        """        
        self.metadata_filename = metadata_filename
        self.metadata = {}
        self.get_metadata()
        
    def get_metadata(self):
        """
        Build the dictionary self.metadata mapping conversation_no to
        dictionaries of values (str, int, or datatime, as
        appropriate).
        """        
        csvreader = csv.reader(open(self.metadata_filename))
        header = next(csvreader)
        for row in csvreader:
            d = dict(list(zip(header, row)))
            for key in ('length', 'from_caller_education', 
                        'to_caller_education'):
                d[key] = int(d[key]) #nb have removed conversation_no
            for key in ('talk_day', 'from_caller_birth_year', 
                        'to_caller_birth_year'):
                #original:
                #d[key] = dateutil.parser.parse(d[key])
                d[key] = int(d[key])
            self.metadata[d['conversation_no']] = d

    def __getitem__(self, val):
        """
        Val should be a key in self.metadata; returns the
        corresponding value.
        """
        return self.metadata[val]

######################################################################

class CorpusReader:
    """Class for reading in the corpus and iterating through its values."""
    
    def __init__(self, src_dirname, metadata_path="swda-metadata.csv"):
        """
        Reads in the data from src_dirname (should be the root of the
        corpus).  Assumes that the metadata file swda-metadata.csv is
        in the main directory of the corpus, using that file to build
        the Metadata object used throughout.
        """
        self.src_dirname = src_dirname
        if not metadata_path == None:
            metadata_filename = os.path.join(src_dirname, metadata_path)
            self.metadata = Metadata(metadata_filename)
        else:
            self.metadata = None

    def iter_transcripts(self, display_progress=True):
        """
        Iterate through the transcripts.

        Argument: 
        display_progress (boolean) -- display an overwriting progress bar if 
        True (default: True)
        """
        i = 1
        #Always src_dirname/*/...utt.csv
        for filename in sorted(iglob(os.path.join(self.src_dirname, "sw*",
                                                   "*utt.csv"))):
            print(filename)
            if "metadata" in filename: continue
            #print "scanning file:",filename
            # Optional progress bar:
            if display_progress:
                sys.stderr.write("\r")
                sys.stderr.write("transcript %s" % i)
                sys.stderr.flush(); i += 1
            # Yield the Transcript instance:
            yield Transcript(filename, self.metadata)
        # Closing blank line for the progress bar:
        if display_progress: sys.stderr.write("\n") 
    
    def get_transcript(self, filename):
        return Transcript(filename, self.metadata)
                    
    def iter_utterances(self, display_progress=True):
        """
        Iterate through the utterances.

        Argument: 
        display_progress (boolean) -- display an overwriting progress 
        bar if True (default: True)
        """
        i = 1
        for trans in self.iter_transcripts(display_progress=False):
            for utt in trans.utterances:
                # Optional progress bar.
                if display_progress:
                    sys.stderr.write("\r")
                    sys.stderr.write("utterance %s" % i)
                    sys.stderr.flush(); i += 1
                # Yield the Utterance instance:
                yield utt
        # Closing blank line for the progress bar:
        if display_progress: sys.stderr.write("\n") 

######################################################################

class Transcript:
    """
    Transcript instances are basically just containers for lists of
    utterances and transcript-level metadata, accessible via
    attributes.
    """
    def __init__(self, swda_filename, metadata):
        """
        Sets up all the attribute values:
	
        Arguments:
        
        swda_filename -- the filename for this transcript
        metadata -- if a string, then assumed to be the metadata
        fileame, and the metadata is created from that filename if a
        Metadata object, then used as the needed metadata directly.        
        """
        self.swda_filename = swda_filename
        # If the supplied value is a filename:
        if isinstance(metadata, str) or isinstance(metadata, str):
            self.metadata = Metadata(metadata)        
        else: # Where the supplied value is already a Metadata object or None.
            self.metadata = metadata
        # Get the file rows:
        rows = list(csv.reader(file(self.swda_filename)))
        #print rows
        # Ge the header and remove it from the rows:
        self.header = rows[0]
        rows.pop(0)
        # Extract the conversation_no to get the meta-data. Use the
        # header for this in case the column ordering is ever changed:
        row0dict = dict(list(zip(self.header, rows[1])))
        #changing from int to string
        #self.conversation_no = int(row0dict['conversation_no'])
        self.conversation_no = row0dict['conversation_no']
        # The ptd filename in the right format for the current OS:
        self.ptd_basename =  os.sep.join(row0dict['ptb_basename'].split("/"))
        # The dictionary of metadata for this transcript:
        if not self.metadata == None:
            transcript_metadata = self.metadata[self.conversation_no]
            for key, _ in transcript_metadata.items():
                setattr(self, key, transcript_metadata[key])
        else:
            transcript_metadata = None
        # Create the utterance list:
        self.utterances = list(map((lambda x : Utterance(x, transcript_metadata)), 
                              rows))
        # Coder's Manual: ``We also removed any line with a "@" 
        #(since @ marked slash-units with bad segmentation).''
        self.utterances = list(filter((lambda x : not re.search(r"[@]", x.act_tag)),
                                  self.utterances))
    
    def next_utt(self, myUtt):
        for utt in self.utterances:
            if utt.transcript_index > myUtt.transcript_index:
                return utt                
        #print("WARNING: No next utterance for FILE " + \
        #str(myUtt.swda_filename) + ", UTT number: " +\
        # str(myUtt.transcript_index))
        return None
    
    def next_utt_same_speaker(self, myUtt):
        for utt in self.utterances:
            if (utt.transcript_index > myUtt.transcript_index 
                and utt.caller == myUtt.caller):
                return utt                
        #print("WARNING: No next utterance same speaker for FILE " 
        #+ str(myUtt.swda_filename) + ", UTT number: " + \
        #str(myUtt.transcript_index))
        return None
    
    def previous_utt_same_speaker(self, myUtt):
        for utt in self.utterances:
            if (self.next_utt_same_speaker(utt) is myUtt):
                return utt
        #print("WARNING: No previous utterance same speaker for FILE " +
        # str(myUtt.swda_filename) + ", UTT number: " +
        # str(myUtt.transcript_index))
        return None
    
    def has_trees(self):
        for utt in self.utterances:
            if hasattr(utt,"trees") and utt.trees != []:
                return True
        return False
    
    def has_pos(self):
        for utt in self.utterances:
            if hasattr(utt,"pos") and utt.pos.strip() != "":
                return True
        return False

            
                
######################################################################
            
class Utterance:
    """
    The central object of interest. The attributes correspond to the
    values of the class variable header:

    'swda_filename':       (str) The filename: directory/basename
    'ptb_basename':        (str) The Treebank filename: add ".pos" for POS and 
                                    ".mrg" for trees
    'conversation_no':     (int) The conversation Id, to key into the 
                                    metadata database.
    'transcript_index':    (int) The line number of this item in the transcript 
                                    (counting only utt lines).
    'act_tag':             (list of str) The Dialog Act Tags (separated 
                                    by ||| in the file).
    'caller':              (str) A, B, @A, @B, @@A, @@B
    'utterance_index':     (int) The encoded index of the utterance 
                                    (the number in A.49, B.27, etc.)
    'subutterance_index':  (int) Utterances can be broken across line. 
                                    This gives the internal position.
    'text':                (str) The text of the utterance
    'pos':                 (str) The POS tagged version of the utterance, 
                                    from PtbBasename+.pos
    'trees':               (list of nltk.tree.Tree) The tree(s) containing 
                                    this utterance (separated by ||| 
                                    in the file).
    'ptb_treenumbers':     (list of int) The tree numbers in the 
                                    PtbBasename+.mrg
    """

    header = [
        'swda_filename',      # (str) The filename: directory/basename
        'ptb_basename',       # (str) The Treebank filename: add ".pos" for 
                              #  POS and ".mrg" for trees
        'conversation_no',    # (int) The conversation Id, to key into the 
                              #  metadata database.
        'transcript_index',   # (int) The line number of this item in the 
                              #  transcript (counting only utt lines).
        'act_tag',            # (list of str) The Dialog Act Tags (separated 
                              #  by ||| in the file).
        'caller',             # (str) A, B, @A, @B, @@A, @@B
        'utterance_index',    # (int) The encoded index of the utterance 
                              # (the number in A.49, B.27, etc.)
        'subutterance_index', # (int) Utterances can be broken across line. 
                              # This gives the internal position.
        'text',               # (str) The text of the utterance
        'pos',                # (str) The POS tagged version of the utterance, 
                              #from PtbBasename+.pos
        'trees',              # (list of nltk.tree.Tree) The tree(s) containing 
                              #this utterance (separated by ||| in the file).
        'ptb_treenumbers'     # (list of int) The tree numbers in the 
                              # PtbBasename+.mrg
       ]
    
    def __init__(self, row, transcript_metadata):
        """
        Arguments:
        row (list) -- a row from one of the corpus CSV files
        transcript_metadata (dict) -- a Metadata value based 
        on the current conversation_no
        """        
        ##################################################
        # Utterance data:
        for i in range(len(Utterance.header)):
            att_name = Utterance.header[i]
            row_value = None
            if i < len(row):
                row_value = row[i].strip()
            # Special handling of non-string values.
            if att_name == "trees":
                #NB CHANGING from below:
                # if row_value: row_value = map(Tree, row_value.split("|||"))
                #print row_value
                if row_value: 
                    trees = []
                    for r in row_value.split("|||"):
                        trees.append(Tree.fromstring(r))
                    #row_value = map(Tree, row_value.split('|||'))
                    row_value = trees
                else: row_value = []
            elif att_name == "ptb_treenumbers":
                if row_value: row_value = list(map(int, row_value.split("|||")))
                else: row_value = []
            elif att_name == 'act_tag':
                # I thought these conjoined tags were meant to be split.
                # The docs suggest that they are single tags, thought,
                # so skip this conditional and let it be treated as a str.
                # row_value = re.split(r"\s*[,;]\s*", row_value)
                # `` Transcription errors (typos, obvious mistranscriptions) 
                #are marked with a "*" after the discourse tag.''
                # These are removed for this version.
                row_value = row_value.replace("*", "")
            elif att_name in ('transcript_index', 'utterance_index', 
                              'subutterance_index'):
                row_value = int(row_value) #NB have removed conversation_no
            # Add the attribute.
            setattr(self, att_name, row_value)
        ##################################################
        # Caller data:
        if not transcript_metadata == None:
            for key in ('caller_sex', 'caller_education', 'caller_birth_year',
                         'caller_dialect_area'):
                full_key = 'from_' + key
                if self.caller.endswith("B"):
                    full_key = 'to_' + key
                setattr(self, key, transcript_metadata[full_key])
        else:
            pass #TODO default values or will this throw a key error?


    def damsl_act_tag(self):
        """
        Seeks to duplicate the tag simplification described at the
        Coders' Manual: 
        http://www.stanford.edu/~jurafsky/ws97/manual.august1.html
        """
        d_tags = []
        tags = re.split(r"\s*[,;]\s*", self.act_tag)
        for tag in tags:
            if tag in ('qy^d', 'qw^d', 'b^m'): pass
            elif tag == 'nn^e': tag = 'ng'
            elif tag == 'ny^e': tag = 'na'
            else: 
                tag = re.sub(r'(.)\^.*', r'\1', tag)
                tag = re.sub(r'[\(\)@*]', '', tag)            
                if tag in ('qr', 'qy'):             tag = 'qy'
                elif tag in ('fe', 'ba'):           tag = 'ba'
                elif tag in ('oo', 'co', 'cc'):     tag = 'oo_co_cc'
                elif tag in ('fx', 'sv'):           tag = 'sv'
                elif tag in ('aap', 'am'):          tag = 'aap_am'
                elif tag in ('arp', 'nd'):          tag = 'arp_nd'
                elif tag in ('fo','o', 'fw', 
                         '"', 'by', 'bc'):          tag = 'fo_o_fw_"_by_bc'            
            d_tags.append(tag)
        # Dan J says (p.c.) that it makes sense to take the first;
        # there are only a handful of examples with 2 tags here.
        return d_tags[0]

    def tree_is_perfect_match(self):
        """
        Returns True if self.trees is a singleton that perfectly matches
        the words in the utterances (with certain simplifactions to each
        to accommodate different notation and information).
        """
        if len(self.trees) != 1:
            return False
        tree_lems = self.regularize_tree_lemmas()
        pos_lems = self.regularize_pos_lemmas()
        if pos_lems == tree_lems:
            return True
        else:
            return False
                                       
    def regularize_tree_lemmas(self):
        """
        Simplify the (word, pos) tags asssociated with the lemmas for
        this utterances trees, so that they can be compared with those
        of self.pos. The output is a list of (string, pos) pairs.
        """        
        tree_lems = self.tree_lemmas()
        tree_lems = list(filter((lambda x : x[1] not in ('-NONE-', '-DFL-')), 
                           tree_lems))
        tree_lems = list(map((lambda x : (re.sub(r"-$", "", x[0]), x[1])), 
                        tree_lems))
        return tree_lems

    def regularize_pos_lemmas(self):
        """
        Simplify the (word, pos) tags asssociated with self.pos, so
        that they can be compared with those of the trees. The output
        is a list of (string, pos) pairs.
        """ 
        pos_lems = self.pos_lemmas()
        pos_lems = list(filter((lambda x : len(x) == 2), pos_lems))
        pos_lems = list(filter((lambda x : x), pos_lems))
        nontree_nodes = ('^PRP^BES', '^FW', '^MD', '^MD^RB', '^PRP^VBZ',
                          '^WP$', '^NN^HVS',
                         'NN|VBG', '^DT^BES', '^MD^VB', '^DT^JJ', '^PRP^HVS',
                          '^NN^POS',
                         '^WP^BES', '^NN^BES', 'NN|CD', '^WDT', '^VB^PRP')        
        pos_lems = list(filter((lambda x : x[1] not in nontree_nodes), pos_lems))
        pos_lems = list(filter((lambda x : x[0] != "--"), pos_lems))
        pos_lems = list(map((lambda x : (re.sub(r"-$", "", x[0]), x[1])), pos_lems))
        return pos_lems
        
    def text_words(self, filter_disfluency=False):
        """
        Tokenized version of the utterance; filter_disfluency=True
        will remove the special utterance notation to make the results
        look more like printed text. The tokenization itself is just
        spitting on whitespace, with no other simplification. The
        return value is a list of str instances.
        """
        t = self.text
        if filter_disfluency:
            t = re.sub(r"([+/\}\[\]]|\{\w)", "", t)
        return re.split(r"\s+", t.strip())

    def pos_words(self, wn_lemmatize=False):
        """
        Return the words associated with self.pos. wn_lemmatize=True
        runs the WordNet stemmer on the words before removing their
        tags.
        """
        lemmas = self.pos_lemmas(wn_lemmatize=wn_lemmatize)
        return [x[0] for x in lemmas]

    def tree_words(self, wn_lemmatize=False):
        """
        Return the words associated with self.trees
        terminals. wn_lemmatize=True runs the WordNet stemmer on the
        words before removing their tags.
        """
        lemmas = self.tree_lemmas(wn_lemmatize=wn_lemmatize)
        return [x[0] for x in lemmas]

    def pos_lemmas(self, wn_format=False, wn_lemmatize=False):
        """
        Return the (string, pos) pairs associated with
        self.pos. wn_lemmatize=True runs the WordNet stemmer on the
        words before removing their tags. wn_format merely changes the
        tags to wn_format where possible.
        """
        pos = self.pos
        pos = pos.strip()
        word_tag = list(map((lambda x : tuple(x.split("/"))), re.split(r"\s+", pos)))
        word_tag = list(filter((lambda x : len(x) == 2), word_tag))
        word_tag = self.wn_lemmatizer(word_tag, wn_format=wn_format,
                                       wn_lemmatize=wn_lemmatize)
        return word_tag        
                
    def tree_lemmas(self, wn_format=False, wn_lemmatize=False):
        """
        Return the (string, pos) pairs associated with self.trees
        terminals. wn_lemmatize=True runs the WordNet stemmer on the
        words before removing their tags. wn_format merely changes the
        tags to wn_format where possible.
        """
        word_tag = []
        for tree in self.trees:
            word_tag += tree.pos()
        return self.wn_lemmatizer(word_tag, wn_format=wn_format, 
                                  wn_lemmatize=wn_lemmatize)
    
    def wn_lemmatizer(self, word_tag, wn_format=False, wn_lemmatize=False):
        # Lemmatizing implies converting to WordNet tags.
        if wn_lemmatize:
            word_tag = list(map(self.__treebank2wn_pos, word_tag))
            word_tag = list(map(self.__wn_lemmatize, word_tag))
        # This is tag conversion without lemmatizing.
        elif wn_format:
            word_tag = list(map(self.__treebank2wn_pos, word_tag))            
        return word_tag
    
    def __treebank2wn_pos(self, lemma):
        """
        Internal method for turning a lemma's pos value into one that
        is compatible with WordNet, where possible (else the tag is
        left alone).
        """
        string, tag = lemma
        tag = tag.lower()
        if tag.startswith('v'):
            tag = 'v'
        elif tag.startswith('n'):
            tag = 'n'
        elif tag.startswith('j'):
            tag = 'a'
        elif tag.startswith('rb'):
            tag = 'r'
        return (string, tag)

    def __wn_lemmatize(self, lemma):
        """
        Lemmatize lemma using wordnet.stemWordNetLemmatizer(). Always
        returns a (string, pos) pair.  Lemmatizes even when the tag
        isn't helpful, by ignoring it for stemming.
        """
        string, tag = lemma
        wnl = WordNetLemmatizer()
        if tag in ('a', 'n', 'r', 'v'):
            string = wnl.lemmatize(string, tag)
        else:
            string = wnl.lemmatize(string)
        return (string, tag)

######################################################################

if __name__ == '__main__':
    pass