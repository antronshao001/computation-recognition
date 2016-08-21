import nengo
from nengo import spa
import gensim
import numpy as np
import matplotlib.pyplot as plt

vocab = spa.Vocabulary(400, randomize=False)
print vocab.vectors

corpus = gensim.models.Word2Vec.load_word2vec_format("wiki_en_text.vector", binary=False)

words_list = ['blue', 'red', 'yellow', 'circle', 'square',  'purple', 'white', 'pink', 'green', 'soxoneonta', 'stripes',
              'dragonhood', 'hooped', 'turquoise', 'rotunda', 'rentable', 'enix', 'frontage', 'cupola',
              'triangle', 'sowwah', 'mickewicz', 'manezhnaya', 'hillman', 'llc', 'widett', 'landon']

for word in words_list:
    vocab.add(word.upper(), corpus[word])

dimensions = 400
model = spa.SPA(label="Simple question answering")

with model:
    model.color_in = spa.Buffer(dimensions=dimensions, vocab=vocab)
    model.shape_in = spa.Buffer(dimensions=dimensions, vocab=vocab)
    model.conv = spa.Memory(dimensions=dimensions, subdimensions=4, synapse=0.4, vocab=vocab)
    model.cue = spa.Buffer(dimensions=dimensions, vocab=vocab)
    model.out = spa.Buffer(dimensions=dimensions, vocab=vocab)
    # Connect the buffers
    cortical_actions = spa.Actions('conv = color_in * shape_in', 'out = conv * ~cue')
    model.cortical = spa.Cortical(cortical_actions)

def color_input(t):
    if t < 0.25:
        return 'RED'
    elif t < 0.5:
        return 'BLUE'
    else:
        return '0'

def shape_input(t):
    if t < 0.25:
        return 'CIRCLE'
    elif t < 0.5:
        return 'SQUARE'
    else:
        return '0'

def cue_input(t):
    if t < 0.5:
        return '0'
    sequence = ['0', 'CIRCLE', 'RED', '0', 'SQUARE', 'BLUE']
    idx = int(((t - 0.5) // (1. / len(sequence))) % len(sequence))
    return sequence[idx]

with model:
    model.inp = spa.Input(color_in=color_input, shape_in=shape_input, cue=cue_input)


# Probe the output
with model:
    model.config[nengo.Probe].synapse = nengo.Lowpass(0.03)
    color_in = nengo.Probe(model.color_in.state.output)
    shape_in = nengo.Probe(model.shape_in.state.output)
    cue = nengo.Probe(model.cue.state.output)
    conv = nengo.Probe(model.conv.state.output)
    out = nengo.Probe(model.out.state.output)

sim = nengo.Simulator(model)
sim.run(3.)


# Plot the results of wiki semantic pointer of similar words

plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.plot(sim.trange(), model.similarity(sim.data, cue))
plt.legend(model.get_output_vocab('cue').keys, fontsize='x-small')
plt.ylabel("cue")

plt.subplot(2, 1, 2)
plt.plot(sim.trange(), spa.similarity(sim.data[out], vocab))
plt.legend(model.get_output_vocab('out').keys, fontsize='x-small')
plt.ylabel("Output")
plt.show()

#check data in 250 and 500 from 0 to 2999 to check if conv correspond to any words

#load 10000 common word list from google
with open('google-en.txt', 'r') as f:
    common_list = f.readlines()

post_list = [pre_word.rstrip() for pre_word in common_list if len(pre_word) > 2]

import operator
test_probe = sim.data[conv][250] #words convoluted of red*circle
test_dict = {common_word: np.dot(corpus[common_word], test_probe) for common_word in post_list}
# sort words list with lowest
sorted_test = sorted(test_dict.items(), key=operator.itemgetter(1))
sorted_test.reverse()
print sorted_test[:10]

test_probe = sim.data[conv][500] #words convoluted of blue*square
test_dict = {common_word: np.dot(corpus[common_word], test_probe) for common_word in post_list}
sorted_test = sorted(test_dict.items(), key=operator.itemgetter(1))
sorted_test.reverse()
