
import numpy as np
import gensim
model = gensim.models.Word2Vec.load_word2vec_format("wiki_en_text.vector", binary=False)
# word_list = ["responsible", "dependable", "persistent", "organized", "ambitious", "neat", "punctual", "disciplined",
#              "efficient", "unreliable", "careless", "undisciplined", "unsystematic", "disorganized", "lazy",
#              "impulsive", "lax", "hedonistic", "sociable", "talkative", "active", "optimistic", "energetic", "lively", "outgoing", "affectionate", "shy",
#              "cautious", "introverted", "reserved", "withdrawn", "silent", "inactive", "unsocial", "sober", "aloof",
#              "unenthusiastic", "quiet", "trusting", "cooperative", "forgiving", "helpful", "modest", "tolerant", "courteous", "flexible",
#              "altruistic", "softhearted", "irritable", "rude", "uncooperative", "aggressive", "suspicious", "impolite",
#              "inflexible", "egoistic", "calm", "relaxed", "confident", "secure", "insensitive","hardy", "unemotional", "depressed", "insecure",
#              "moody", "anxious", "hopeful", "encouraging", "hypochondriac", "excitable", "worrisome", "hypersensitive",
#              "nervous", "depressed", "insecure",
#              "moody", "anxious", "hopeful", "encouraging", "hypochondriac", "excitable", "worrisome", "hypersensitive",
#              "nervous"]

word_list = ["curious", "original", "intellectualism", "creative", "narrow", "inartistic", "stubborn", "unoriginal",
             "responsible", "organized", "neat", "efficient", "unsystematic", "disorganized", "lax", "hedonistic",
             "sociable", "active", "lively", "outgoing", "reserved", "withdrawn", "silent", "unsocial", "cooperative",
             "modest", "flexible", "altruistic", "irritable", "rude", "suspicious", "impolite", "relaxed", "secure",
             "hardy", "unemotional", "moody", "encouraging", "hypochondriac", "nervous"]

#word_list = ["sociable", "talkative", "active", "optimistic", "energetic", "irritable", "rude", "uncooperative",
#             "aggressive", "cat"]

vect = [model[word] for word in word_list]

# maximize hypercube space of an enclosing point set
# use dimensional multiplication or norm 2 to calculate hypercube space
def hypercube(vec_list):
    sup_vec = np.amax(vec_list, axis=0)
    inf_vec = np.amin(vec_list, axis=0)
    # result = 1
    # for dim in np.subtract(sup_vec, inf_vec):
    #     result *= dim
    # return result
    return np.linalg.norm(np.subtract(sup_vec, inf_vec))

# calculate hyper sphere radius from hypercube

# push index to the next order
def pushindex(index_list, numb):
    open_list = [i for i in range(numb) if i not in index_list]
    if index_list[0] > open_list[-1]: #stuck
        return False
    elif index_list[-1] < open_list[-1]: # still space for last to push
        temp = index_list.pop(-1)
        index_list.append(temp+1)
        return True
    else: # find the last index with empty space to move
        movable_list = [i for i in index_list if i < open_list[-1]]
        move_index = len(index_list)-len(movable_list)
        temp = movable_list[-1]
        index_list.remove(temp)
        index_list.insert(len(movable_list)-1, temp+1)
        # move last move index to temp+1
        for i in range(move_index):
            index_list.pop(-1)
        index_list.extend(range(numb)[temp+2:temp+2+move_index])
        return True

# selsect orderly from data set to find max
# in here vec_list is a list of number vector to represent a word
def select_vec(vec_list, numb):
    if len(vec_list) <= numb:
        return [vec_list.index(vector) for vector in vec_list]
    else:
        max_index, list_index = range(numb), range(numb)
        max_space = hypercube([vec_list[index] for index in list_index])
        while pushindex(list_index, len(vec_list)):
            # print list_index
            # print hypercube([vec_list[index] for index in list_index])
            if hypercube([vec_list[index] for index in list_index]) > max_space:
                max_index = list(list_index)
                max_space = hypercube([vec_list[index] for index in list_index])
        return max_index


vect = [model[word] for word in word_list]
out_index = select_vec(vect, 4)
print out_index
print [word_list[ind] for ind in out_index]


word_list = ["relaxed", "secure","hardy", "unemotional", "moody", "encouraging", "hypochondriac", "nervous"]