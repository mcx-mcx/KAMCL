import os
import re
import sys
import nltk
import json
import argparse
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from collections import Counter
import vocab

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower()

ENGLISH_STOP_WORDS = set(stopwords.words('english'))
'''
get the labels of each caption on train&val&test  actually only use train labels in train in test/val will use
the labels that creat by retriveed captions
'''
def checkToSkip(filename, overwrite):
    if os.path.exists(filename):
        print ("%s exists." % filename),
        if overwrite:
            print ("overwrite")
            return 0
        else:
            print ("skip")
            return 1
    return 0

def makedirsforfile(filename):
    try:
        os.makedirs(os.path.split(filename)[0])
    except:
        pass

def get_wordnet_pos(tag):
    '''
    This function will get each word speech
    :param tag: have tagged speech of words
    :return:the part of speech of word
    '''
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    else:
        return None


def fromtext(opt, txt):
    image2words = {}
    count = 0               
    with open(txt, 'r') as t:
        for line in tqdm(t.readlines(), desc='Reading and processing captions'):  
            if count % 5 == 0 :
                image2words[str(count // 5)] = []
                cap = clean_str(line)
                image2words[str(count // 5)].extend(cap.strip().split(" "))
            else :
                cap = clean_str(line)
                image2words[str(count // 5)].extend(cap.strip().split(" "))
            count += 1
            
    return image2words  # {vid:[word,word,word....],....}

def get_tags(id2words, threshold, output_file, str_name):
    fout = open(output_file, 'w')
    for imageid in tqdm(id2words.keys(), desc='Processing {0} labels'.format(str_name)):
        word2counter = {}
        fout.write('%s\t' % imageid)

        # if imageid.startswith('tgif'):
        #     new_threshold = 0
        # else:
        new_threshold = threshold

        for w in id2words[imageid]:
            word2counter[w] = word2counter.get(w, 0) + 1  # count word frequency
        word2counter = sorted(word2counter.items(), key=lambda a: a[1],
                              reverse=True)  # return a list like [(word:frequency)]

        for (word, value) in word2counter:
            if value > new_threshold:
                fout.write('%s:%d ' % (word, value))  # (word , frequency)
        fout.write('\n')
    fout.close()

def main(opt):
    rootpath = opt.rootpath
    collection = opt.collection
    threshold_im = opt.threshold_im
    threshold_tx = opt.threshold_tx
    use_lemma = opt.use_lemma
    tag_vocab_size = opt.tag_vocab_size

    output_image_tags = os.path.join(rootpath,'data', '%s_precomp'%collection, 'tags_label_th_%d.txt' %threshold_im  )
    # output_sent_labels = os.path.join(rootpath, collection, "TextData", 'tags',  'sentence_label_th_%d.txt' % (threshold_tx))

    if checkToSkip(output_image_tags, opt.overwrite):
        sys.exit(0)
    makedirsforfile(output_image_tags)

    tag_vocab_dir = os.path.join(rootpath, 'data', '%s_precomp'%collection, 'tags_label_th_%d'  %threshold_im )
    all_words_file = os.path.join(tag_vocab_dir, 'all_words.txt')
    if checkToSkip(all_words_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(all_words_file)

    cap_file = os.path.join(rootpath, 'data','%s_precomp'%collection, 'train_caps.txt')
    if not os.path.exists(cap_file):
        cap_file = os.path.join(rootpath, 'data', '%s_precomp'%collection, 'train_caps.txt')
    image2words= fromtext(opt, cap_file)
 
    get_tags(image2words, threshold_im, output_image_tags, 'image')
    # get_tags(sid2words, threshold_tx, output_sent_labels, 'sentence')
    print ('The image labels have saved to %s' % (output_image_tags))
    # print ('The sentence labels have saved to %s' % (output_sent_labels))

    
    # generate tag vocabulary
    lines = map(str.strip, open(output_image_tags))
    cnt = Counter()

    for line in lines:
        elems = line.split()
        del elems[0]
        # assert(len(elems)>0)
        for x in elems:
            tag,c = x.split(':')
            cnt[tag] += int(c)

    vocabulary = vocab.deserialize_vocab('/home/mcx/RS/KAMCL-main/vocab/nwpu_precomp_vocab.json')

    print(len(cnt))
    taglist = cnt.most_common()
    taglist = [x for x in taglist if x[0] in vocabulary.word2idx.keys()]

    fw = open(all_words_file, 'w')
    fw.write('\n'.join(['%s %d' % (x[0], x[1]) for x in taglist]))
    fw.close()

    top_tag_list = [x[0] for x in taglist]

    # save tag vocabulary
    output_json_file = os.path.join(tag_vocab_dir, 'tag_vocab_%d.json' % tag_vocab_size)
    output_txt_file = os.path.join(tag_vocab_dir, 'tag_vocab_%d.txt' % tag_vocab_size)

    with open(output_json_file, 'w') as jsonFile:
        jsonFile.write(json.dumps(top_tag_list[:tag_vocab_size]))

    with open(output_txt_file, 'w') as txtFile:
        txtFile.write('\n'.join(top_tag_list[:tag_vocab_size]))

    # open(output_file, 'w').write('\n'.join(output_vocab))
    print('Save words to %s and %s' % (output_json_file, output_txt_file))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--rootpath', default='/home/mcx/RS/KAMCL-main-final', help='rootpath of the data')
    parser.add_argument('--collection', default="rsicd", type=str, help='collection')
    parser.add_argument('--threshold_im', type=int, default=1, help='minimum word count threshold for image labels')
    parser.add_argument('--threshold_tx', type=int, default=0, help='minimum word count threshold for sentence labels')
    parser.add_argument('--overwrite', default=1, type=int, help='overwrite existing file (default=0)')
    parser.add_argument('--use_lemma', action="store_true", help='whether use lemmatization')
    parser.add_argument('--tag_vocab_size', default=32, type=int, help='vocabulary size of concepts tags')
    args = parser.parse_args()
    # opt = opts.parse_opt()
    main(args)