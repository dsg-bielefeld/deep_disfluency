# various functions that are used in many places in the scripts

import os.path
import argparse
import sys
import collections
import codecs
import time
import datetime
import multiprocessing
import itertools
import logging

logger_formatting = '%(asctime)s %(filename)s @ %(funcName)s - \
%(levelname)s - %(message)s'
tag_separator = '_@'

def glue_tokens(tokens):
        """The standard way in which we glue together tokens to 
        create keys in our hashmaps"""
        return ' '.join(tokens)

def ngrams(tokens,max_order,ngram_maps):
        """This function counts the ngrams in tokens (a list of things) 
        for each order between 1 and max_order"""
        l = len(tokens)
        for i in range(l):
            for d in range(1,max_order+1):
                ngram_map = ngram_maps[d]
                j=i+d
                if not j > l:
                    key = glue_tokens(tokens[i:j])
                    ngram_map[key] += 1
        return ngram_maps

def get_word(tag_and_word):
    x = tag_and_word.split(tag_separator)
    try:
        return x[1]
    except IndexError:
        logging.warning('{0} is not tagged.\
        This is not a problem now because we are just \
        returning the word.\n'.format(tag_and_word))
        return x[0]

def get_tag(tag_and_word):
    x = tag_and_word.split(tag_separator)
    return x[0]

def get_tag_and_word(tag_and_word):
    x = tag_and_word.split(tag_separator)
    try:
        return (x[0],x[1])
    except IndexError:
        logging.warning('{0} is not tagged. This may be a problem later \
        though.Returning the word as its own tag...\n'.format(tag_and_word))
        return(x[0],x[0])

def safe_open(filename,mode='r'):
    """To avoid forgetting to first expand system
    variables in file names and to chop off a trailing newline 
    it's better to use this function"""
    return open(os.path.expandvars(filename.strip()),mode)

def flush_and_close(f):
    """Flushes and closes the file, really"""
    if not f.closed and not f.isatty():
        f.flush()
        os.fsync(f.fileno())
        f.close()

def safe_open_with_encoding(filename,mode,encoding='utf-8'):
    return codecs.open(os.path.expandvars(filename.strip()),mode,
                       encoding=encoding,errors='backslashreplace')

def plot_LM_values_for_utterance(lm,lm_function,function_name,words):
    """plots the local LM value of interest (e.g. entropy of continuation/WML) 
    of words by the language model, missing first n-1 tokens as context"""
    import matplotlib.pyplot as plt
    delta = lm.order -1
    newwords = []
    j = 0
    plt.close('all')
    for i in range(delta,len(words)): #creates groups of ngrams
        newwords.append(words[i-delta:i+1])
        j+=1
    #get the local value of interest (WML/prob etc)
    results = []
    for word in newwords:
        results.append(lm_function(word))
    #dummy x-axis
    nums = []
    for i in range(0,len(newwords)):
        nums.append(float(i+1))
    plt.plot(nums,results)
    words = words[delta:]
    plt.xticks(nums,words) #replace dummy axis with words
    plt.ylabel(function_name)
    plt.show()



def process_args_for_consumer_producer(*args,**kwargs):
    """Process arguments for a script that work as a 
        regular unix command line tool
       reading input from stdin or a file and writing output 
       to stdout or a file
       The function returns the input stream and the output stream. 
       It is possible to arguments to the parser constructor 
       (for instance the description of the script)
    """
    parser = argparse.ArgumentParser(*args,**kwargs)
    parser.add_argument('-i', '--input', type=file, metavar='INPUT',
            help='read input from INPUT (must be a file) instead of stdin')
    parser.add_argument('-o', '--output', type=str, metavar='OUTPUT',
        help='write output from OUTPUT (must be a file) instead of stdout')
    parser.add_argument('-l', '--logger', type=str, required=False)
    args = parser.parse_args()
    if args.input == None:
        args.input = sys.stdin
    if args.output == None:
        args.output = sys.stdout
    else:
        args.output = safe_open(args.output,'w')
    if not args.logger is None:
	    args.logger = safe_open(args.logger,'a')
    return args



def file_identifier_from_sentence_identifier(sent_id):
    """ Extract the file id from a sentence identifier """
    return sent_id[:3]

def sentence_number_from_sentence_identifier(sent_id):
    """ Extract the sentence number from a sentence identifier """
    return sent_id[4:]

def key_value_list_to_map(list):
    """A function to transform key values lists into maps"""
    m = {}
    for k,v in list:
        m[k] = v
    return m



class FrequencyDistribution(collections.defaultdict):

    def __init__(self,observations=[]):
        collections.defaultdict.__init__(self,int)
        self.total_counts = 0
        self.add_counts(observations)
        
    def to_ordered_list(self,reversed_order=False):
        """
           Returns an ordered version of this frequency distribution

           reversed_order -- if True the frequency are ordered 
           from lowest to highest
        """
        res = []
        for i in sorted(self,key=self.get,reverse=(not reversed_order)):
            res.append((i,self[i]))
        return res

    def add_counts(self,seq):
        """Add counts to the distribution"""
        for i in seq:
            self[i] += 1
            self.total_counts += 1

    def add(self,x):
        """Add a single element to the counts"""
        self[x] += 1
        self.total_counts += 1

    def get_n_most_frequent(self,n):
        res = []
        for i in sorted(self,key=self.get,reverse=True):
            res.append(i)
            n -= 1
            if n <= 0:
                break
        return res

    def pprint(self,output_stream=sys.stdout):
        """Pretty print"""
        for i in sorted(self,key=self.get,reverse=True):
            output_stream.write('{0} {1}\n'.format(i,self[i]))


class CountingList(collections.MutableSequence):
	def __init__(self):
            self.count = 0
            self.list = list()
            
        def __len__(self):
            return len(self.list)

        def __getitem__(self,i):
            return self.list[i]

        def __setitem__(self,i,v):
            self.list[i] = v

        def __delitem__(self,i):
            self.count -= 1
            del self.list[i]

        def insert(self,i,v):
            self.list.insert(i,v)

        def append(self,x):
            self.count += 1
            self.list.append(x)

def is_the_corpus_tagged(stream):
	"""Checks whether a corpus is tagged.

	   To do so, we check if at least one word is
	    tagged in the first sentence. 
	    After that we reset the stream position to the beginning"""
	line = stream.readline()
	# we skip empty lines
	while (line == ''):
            line = stream.readline()
	stream.seek(0)
	tokens = line.split()
	if any([tag_separator in x for x in tokens]):
		return True
	else:
		return False

def add_seed_argument(parser):
    parser.add_argument('-s','--seed', type=int,metavar='SEED',
			    help='the seed used to initialize the number generator. \
			    Define it for reproducibility of results \
			    (it defaults to 5489 in any case)',default=5489)

# statistical stuff

def percentile(values,p):
    """
    Computes the percentile of a list of values using the NIST method.
    
    @parameter values - a list of values
    @parameter p - the percentage, a float value from in the [0.0,100.0) 
                    interval
    
    @return - the percentile
    """
    sorted_values = sorted(values)
    N = len(sorted_values) - 1
    rank = p / 100. * (N + 1)
    k = int(rank)
    d = rank - k
    if k == 0:
	    return sorted_values[0]
    elif k == N:
	    return sorted_values[-1]
    else:
	    return sorted_values[k-1] + d * (sorted_values[k] - sorted_values[k-1])

def read_property_file(f):
    """Reads a properties file"""
    props = dict()
    for line in f:
        content = line.split('#')[0].strip()
        if not content == '':
            key_and_val = content.split('=')
            key = key_and_val[0].strip()
            val = '='.join(key_and_val[1:])
            if val[0] == '"' and val[-1] == '"':
                val = val[1:-1]
            elif val[0] == "'" and val[-1] == "'":
                val = val[1:-1]
            props[key] = val.strip()
    return props


def pointwiseFold(f,x,n,listOfLists):
    """This function is a generalization of the fold function 
    to take more than one list.

       Keyword arguments:
       f -- the binary function used to accumulate some value
       x -- the initial value for all the values
       n -- the common length of the lists
       listOfLists -- what it says
    """
    res = [x] * n
    for l in listOfLists:
        for i in range(n):
            res[i] = f(res[i],l[i])
    return res

def sync_parallel_map_file(f,file,pool=None,n_processes=None,timer=None):
    if n_processes is None:
        try:
            n_processes = multiprocessing.cpu_count()
        except NotImplementedError:
            n_processes = 2

    if pool is None:
        pool = multiprocessing.Pool(processes=n_processes)
        
    for lines in itertools.zip_longest(*[file]*n_processes):
        pool.map(f,[x for x in lines if not x is None])
        if not timer is None:
            for i in range(n_processes):
                timer.advance()

def concurrent_process_file(f,file,pool=None):
    if pool is None:
        pool = multiprocessing.Pool()
    res = pool.map(f,file)
    pool.close()
    pool.join()
    return res

class Timer(object):
    """An object that logs the progress of a process and estimates 
    the remaining time to its end"""
    def __init__(self,total_number_of_steps,log_stream=sys.stderr):
        self.total_number_of_steps = total_number_of_steps
        self.log_stream = log_stream
        self.current_step = 0.
        self.start_time = None
        self.previous_message_length = 1
        
    def advance(self):
        if self.start_time is None:
            self.start_time = time.time()
        self.current_step += 1.
        current_percentage = self.current_step / \
                    self.total_number_of_steps * 100.
        current_time = time.time()
        estimated_completion_time = (current_time - self.start_time) * \
        (100. / current_percentage) - (current_time - self.start_time)
        msg = '{:>6.3f}% completed -- Estimated remaining time: {:}\r'.\
        format(current_percentage,str(
                                datetime.timedelta(
                                    seconds=estimated_completion_time)))
        self.log_stream.write(' ' * self.previous_message_length + '\r')
        self.log_stream.write(msg)
        self.previous_message_length = len(msg)

    def done(self):
        self.log_stream.write('\n')
        

def collect_ngrams(corpus_file,max_order,glue_tokens,tokenize_sentence):
    """Returns the set of all ngrams up to max_order that appear in the file, 
    for convenience we also count the number of lines..."""
    ngrams = set()
    n_lines = 0
    for line in corpus_file:
        n_lines += 1
        tokens = tokenize_sentence(line)
        l = len(tokens)
        for i in range(l):
            for d in range(1,max_order+1):
                j=i+d
                if not j > l:
                    ngrams.add(glue_tokens(tokens[i:j],d))
    return ngrams,n_lines

def is_xml_file(f):
    """Tries to guess if the file is a BNC xml file"""
    f.seek(0)
    line = f.readline().strip()
    while line == '':
        line = f.readline().strip()
    if line.find('<bncDoc ') != -1:
        return True
    else:
        return False

def interleave(list1,list2):
    if list1 == []:
        return list2
    elif list2 == []:
        return list1
    else:
        return [list1[0]] + [list2[0]] + interleave(list1[1:],list2[1:])
        
