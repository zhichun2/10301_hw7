############ Welcome to HW7 ############
# TODO: Andrew-id: zhichun2


# Imports
# Don't import any other library
import argparse
import numpy as np
from utils import make_dict, parse_file
import logging

# Setting up the argument parser
# don't change anything here

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to store the hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to store the hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to store the hmm_transition.txt (B) file')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


# Hint: You might find it useful to define functions that do the following:
# 1. Calculate the init matrix
# 2. Calculate the emission matrix
# 3. Calculate the transition matrix
# 4. Normalize the matrices appropriately

# TODO: Complete the main function
def main(args):

    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)
    print(word_dict)
    # print(tag_dict)

    # Parse the train file
    # Suggestion: Take a minute to look at the training file,
    # it always helps to know your data :)
    sentences, tags = parse_file(args.train_input)

    logging.debug(f"Num Sentences: {len(sentences)}")
    logging.debug(f"Num Tags: {len(tags)}")
    
    
    # Train your HMM
    # print(sentences)
    # print(tags)
    # convert tag 2d array into 1d numpy array
    np_tags = np.array(tags[0])
    for i in range(1, len(tags)):
        temp = np.array(tags[i])
        np_tags = np.append(np_tags, temp)
    
    # convert sentences 2d array into 1d numpy array
    np_sentences = np.array(sentences[0])
    for i in range(1, len(sentences)):
        temp = np.array(sentences[i])
        np_sentences = np.append(np_sentences, temp)  

    # init matrix
    # an array of the initial tags
    init = []
    for l in tags:
        init.append(l[0])
    np_init = np.array(init)        

    tag_occurrences = np.zeros(len(tag_dict))
    i = 0
    for key in tag_dict:
        tag_occurrences[i] = np.sum(np_init == key) + 1
        i += 1
    tag_denominator = np_init.shape[0] + len(tag_dict) 
    init = np.true_divide(tag_occurrences, tag_denominator)
    # print(init)

    # transition matrix
    trans_count = np.zeros((len(tag_dict), len(tag_dict)))
    for i in range(len(tags)):
        line = tags[i]
        for j in range(len(line)-1):
            row = tag_dict[line[j]]
            col = tag_dict[line[j+1]]
            trans_count[row][col] += 1
    trans_pesudocount = trans_count + 1
    trans_sum = np.sum(trans_pesudocount, axis=1)
    transition = trans_pesudocount / trans_sum[:,None]
    # print(transition)

    # emission matrix
    match = np.vstack((np_tags, np_sentences))
    emit_count = np.zeros((len(tag_dict), len(word_dict)))
    for i in range(len(match[0])):
        row = tag_dict[match[0][i]]
        col = word_dict[match[1][i]]
        emit_count[row][col] += 1
    emit_pesudocount = emit_count + 1
    emit_sum = np.sum(emit_pesudocount, axis=1)
    emission = emit_pesudocount / emit_sum[:,None]
    # print(emission)



    # Making sure we have the right shapes
    # logging.debug(f"init matrix shape: {init.shape}")
    # logging.debug(f"emission matrix shape: {emission.shape}")
    # logging.debug(f"transition matrix shape: {transition.shape}")


    ## Saving the files for inference
    ## We're doing this for you :)
    ## TODO: Just Uncomment the following lines when you're ready!
    
    np.savetxt(args.init, init)
    np.savetxt(args.emission, emission)
    np.savetxt(args.transition, transition)

    return 

# No need to change anything beyond this point
if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)