
import numpy as np
import gensim
model = gensim.models.Word2Vec.load_word2vec_format("wiki_en_text.vector", binary=False)

# use the model to do word logic inference: france as to paris = taiwan as to taipei
comparison = 'france'
asto = 'paris'
find_asto = 'taiwan'
print model.most_similar(positive=[find_asto,asto], negative=[comparison])

#use model to find 10 most related words
find_similar = 'baby'
relattion_numb = 10
print model.most_similar(find_similar, topn=relattion_numb)