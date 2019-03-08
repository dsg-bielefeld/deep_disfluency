import re
from collections import defaultdict
from difflib import SequenceMatcher


def cleanup(astring):
    """this allows us to do agreement per word (without weird characters)"""
    #i.e. if this returns "" then don't consider it..
    astring = re.sub("<[^>]+>", "", astring) #gets rid of tags
    unwantedchars = ['{', '{f', '{F', '}', '.', '+', '(', ')', '\'', ',', '-', '$unc$']
    for achar in unwantedchars:
        astring = astring.replace(achar, "")
    #astring = astring.strip()
    #astring = " ".join(astring.split())
    return astring

def align(string1,string2):
    """min edit distance string aligner for string2-> string1
    Creates table of all possible edits, only considers the paths from i,j=0 to i=m, j= n
    returns mapping from i to j with the max alignment- problem, there may be several paths. Weights:
        identity(string1,string2) 0
        insert(eps,string) 1
        delete(string1,eps) 1
        substitution(string1,string2) 2
    """
    #we have the normal del, insert and subs for min edit distance
    #we only need define the sub relation for the special cases
    #pointers
    
    #do lower case for all
    string1 = [x.lower().replace("<laughter>","").\
                  replace("</laughter>","") for x in string1]
    string2 = [x.lower().replace("<laughter>","").\
                  replace("</laughter>","") for x in string2]
    
    
    left = "<"
    up = "^"
    diag = "\\"
    m = len(string1)
    n = len(string2)
    string1 = ["",] + list(string1) #add the empty strings
    string2 = ["",] + list(string2)
    #print "initial:"
    #print string1
    #print string2
    
    def sub(source,goal,initial): #2 words #initial = the table's content initialised as [i,j,"",currentScore,""]
        if source == goal: return initial[0:2] + ["ID"] + [initial[3],diag]#NO COST
        elif cleanup(source) == cleanup(goal): return initial[0:2] + ["ID"] + [initial[3],diag]#NO COST ?
        else: return initial[0:2] + ["S_ARB"] + [initial[3]+2,diag] #2 for arbitrary sub
    
    def delete(source,initial):
        category = "DEL"
        return initial[0:2] + ["DEL"] + [initial[3]+1,up]
    
    def insert(goal,initial):
        category = "INS"
        return initial[0:2] + ["INS"] + [initial[3]+1,left]
        
    #initilisation of axes in table, hash from number to number to list (cell)
    D = [] #the cost table
    ptr = [] #the pointer table
    for i in range(0,m+1):
        D.append([0]*(n+1))
        ptr.append([[]]*(n+1)) #these are mutable, just dummies
    
    #defaultdict(defaultdict(list)) #the pointer table with a list of (pointer,relation) pairs
    #populate each of the table axes
    D[0][0] = 0
    j = 0
    for i in range(1,m+1):
        a = delete(string1[i],[i,j,"",D[i-1][j],""])
        D[i][j] = a[3] #delete cost
        ptr[i][j] = [(a[-1],a[2])] #delete type
    i = 0
    for j in range(1,n+1):
        a = insert(string2[j],[i,j,"",D[i][j-1],""])
        D[i][j] = a[3] #insert cost
        ptr[i][j] = [(a[-1],a[2])] #insert type
    
    #for i in range(0,m+1):
    #    print D[i]
        
    #for i in range(0,m+1):
    #    print ptr[i]
    
    #main recurrence relation algorithm
    for i in range(1,m+1):
        for j in range(1,n+1):
            #print "%%%%%%%"
            #print i
            #print j
            deltest = delete(string1[i],[i,j,"",D[i-1][j],""])
            #print deltest
            instest = insert(string2[j],[i,j,"",D[i][j-1],""])
            #print instest
            subtest = sub(string1[i],string2[j],[i,j,"",D[i-1][j-1],""])
            #print subtest
            #print "%%%%%%%"
            #get the min cost set
            mincostset = set()
            mincostset.add(tuple(deltest))
            mincost = deltest[-2]
            tests = [instest,subtest] #check the others
            for t in tests:
                if t[-2] < mincost:
                    mincost = t[-2]
                    mincostset = set()
                    mincostset.add(tuple(t))
                elif t[-2] == mincost:
                    mincostset.add(tuple(t))
            #add the pointers and their alignments
            ptr[i][j] = []
            for a in mincostset:
            #    print a
                ptr[i][j].append((a[-1],a[2]))
            D[i][j] = mincost
    #print the optimal alignment(s) backtrace-
    #there should only be one given the weights as we shouldn't allow an ins+del to beat an arbsub
    #return a list of the alignemnts
    #gets them backwards then returns the reverse
    #print "cost = " + str(D[m][n])
    #for i in range(0,m+1):
    #    print D[i]
        
    #for p in range(0,m+1):
    #    print ptr[p]
    
    #return all and rank by best first approach
    #if there is a branch, follow and pop the first pointer, effectively removing the path
    def backtrace(D,ptr,i,j,mymap,mymaps):
        if i == 0 and j == 0: #should always get there directly
            mymaps.append(mymap)
            return
        arrow = ptr[i][j][0][0] #get the first one
        alignment = ptr[i][j][0][1]
        score = D[i][j]
        if len(ptr[i][j])>1: #more than one!
            del ptr[i][j][0] #remove it before copying and recursing
            #mymapcopy = list(mymap)
            backtrace(D,ptr,i,j,list(mymap),mymaps)
            #ptr[i][j] = filter(lambda x: not x[0] == "\\", ptr[i][j])
            #coarse approximation
        mymap.insert(0,tuple([max(0,i-1),max(0,j-1),alignment,score]))
        
        if arrow == "\\":
            backtrace(D,ptr,i-1,j-1,mymap,mymaps)
        elif arrow == "^":
            backtrace(D,ptr,i-1,j,mymap,mymaps)
        elif arrow == "<":
            backtrace(D,ptr,i,j-1,mymap,mymaps)
        
    def rank(mymaps,start,n):
        tail = []
        for j in range(start,n):
            bestscores = []
            if len(mymaps) == 1: return mymaps + tail
            for mymap in mymaps: #should this recurse to the last mapping to j (i.e. highest value for i)? yes
                for mapping in mymap:
                    if mapping[1] == j:
                        bestscore = mapping[3]
                    elif mapping[1] >j: break
                bestscores.append(bestscore) #should always get one!
            best = min(bestscores)
            #print "best"
            #print best
            #maintain all the best for further sorting; separately sort the tail?
            i = 0; a = 0
            while i < len(bestscores):
            #    print bestscores[i]
                if bestscores[i] > best:
                    tail.append(list(mymaps[a])) #bad score
                    del mymaps[a]
                else: a+=1
                i+=1
            if len(tail)>0: tail = rank(tail,j,n) #recursively sort the tail
        #print "warning no difference!!"
        return mymaps #if no difference just return all        
    
    mymaps = []
    mymap = []
    backtrace(D,ptr,m,n,mymap,mymaps)
    if len(mymaps)>1:
        #print "ranking"
        #print len(mymaps)
        #print mymaps
        mymaps = rank(mymaps,0,n) #sorts the list by best first as you pass left to right in the repair
    #for mapping in mymaps:
    #    print mapping
    #print "returning:"
    #print mymaps[0]
    return mymaps[0] #only returns top, can change this

def matchBlock(ss):
    """"Returns the best alignments from each sequence to its closet matching one and the matching
    character blocks for each match. If there are 3 sequences (expected), then it need 
    only return the best two alignments. The direction of alignment will be outputted too.
    In using these one can reverse the order if there is no link from a given sequence to the next"""
    #step 1 do pairwise sequence matching on all in ss and store alignments and scores
    s = SequenceMatcher()
    matchstore = []
    for i in range(len(ss)):
        #print "********"
        #print ss[i]
        no_1 = i
        x = ss[i].lower()
        s.set_seq1(x)
        for j in range(0,len(ss)):
            if j == i: continue # do not align to self
            no_2 = j
            #print ss[j]

            y = ss[j].lower()
            s.set_seq2(y)
            #print s.ratio()
            #print s.get_matching_blocks()
            matchstore.append((no_1,no_2,s.ratio(),tuple(s.get_matching_blocks())))
    bestlist = sorted(matchstore,key=lambda x: x[2], reverse=True)  #step 2 find best match in that pairwise agreement
    best = bestlist[0]
    #print best
    bestset = set()
    bestset.add(tuple(best))
    remainder = set(range(len(ss)))  #step 3 align the remaining sequence to the sequence it matches best to to that sequence
    remainder.remove(best[0])
    remainder.remove(best[1])
    for i in remainder: # should only be one
        for b in bestlist[1:]:
            if i in [b[0],b[1]]:
                if (best[0] in [b[0],b[1]] or best[1] in [b[0],b[1]]):
                    #this is the best alignment involving this one
                    bestset.add(b)
                    break
    bestMatches = defaultdict()
    for b in bestset:
        bestMatches[(b[0],b[1])] = b[2]
    
    
if __name__ == '__main__':
    #s = SelfRepair()
    string1= ["I", "like", "john"]
    string2 = ["<laughter/>I","like","john"]
    print(align(string2,string1))
    #===========================================================================
    # a = "<breathing/> Wo man richtig geil drauf chillen kann"
    # b = "Wo man richtig geil drauf chillen kann"
    # c = "wo man richtig geil drauf chillen kann"
    # ss = [a,b,c]
    # print matchBlock(ss)
    #===========================================================================