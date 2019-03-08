# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET  # todo cElementTree
import os
import codecs

word_count = 0


def safe_open(filename, mode='r'):
    """To avoid forgetting to first expand system
    variables in file names and to chop off a trailing newline 
    it's better to use this function.
    """
    return open(os.path.expandvars(filename.strip()), mode)


def safe_open_with_encoding(filename, mode, encoding='utf-8'):
    return codecs.open(os.path.expandvars(filename.strip()), mode, 
                       encoding=encoding, errors='backslashreplace')


def readXML_writeCorpus(orig_xml, new_file):
    global word_count
    # do the xml magic
    tree = ET.parse(orig_xml)
    #tree = None
    dom = tree.getroot()
    newdom = ET.Element('text')
    #newdom = None
    #count = 30
    for s in dom.iter('s'):
        tokens = []
        # count-=1
        # if count ==0 : break
        newdom.append(s)
        for w in s.iter('w'):
            # print w.text, w.attrib['c5'] #POS tag
            tokens.append(
                "_@".join([w.attrib['c5'].strip(), 
                           w.text.encode('utf8').strip()]))
        new_file.write(" ".join(tokens) + "\n")
        word_count += len(tokens)

    orig_xml.close()


def main():
    list_file = "../data/raw_data/bnc_spoken/ranges/\
    demographic_contextdependent.txt"
    # create new corpus file
    new_file = open(
        "../data/raw_data/bnc_spoken/BNC_spoken_tags-and-words.CORPUS", "w")
    # get the filenames
    bnc_files = [line.strip() for line in open(list_file)]
    bnc_dir = "/Volumes/My Passport/bielefeld-server/\
    External/BNC/BNC-XML/Texts"
    for filename in bnc_files:
        try:
            # load original bnc corpus file
            orig_xml = safe_open(os.path.join(bnc_dir, filename + ".xml"), 'r')
        except IOError:
            print("NO file", filename + ".xml")
            continue
        readXML_writeCorpus(orig_xml, new_file)  # write to new file
        print(word_count)
        if word_count > 20000000:
            break
    new_file.close()

if __name__ == '__main__':
    main()
