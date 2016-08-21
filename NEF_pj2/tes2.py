
import matplotlib.pyplot as plt

import nengo
from nengo import spa


dimensions = 400

model = spa.SPA(label="Simple question answering")

with model:
    model.color_in = spa.Buffer(dimensions=dimensions)
    model.shape_in = spa.Buffer(dimensions=dimensions)
    model.conv = spa.Memory(dimensions=dimensions, subdimensions=4, synapse=0.4)
    model.cue = spa.Buffer(dimensions=dimensions)
    model.out = spa.Buffer(dimensions=dimensions)
    
    # Connect the buffers
    cortical_actions = spa.Actions(
        'conv = color_in * shape_in',
        'out = conv * ~cue'
    )
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


# ## Probe the output

# In[ ]:

with model:
    model.config[nengo.Probe].synapse = nengo.Lowpass(0.03)
    color_in = nengo.Probe(model.color_in.state.output)
    shape_in = nengo.Probe(model.shape_in.state.output)
    cue = nengo.Probe(model.cue.state.output)
    conv = nengo.Probe(model.conv.state.output)
    out = nengo.Probe(model.out.state.output)


# ## Run the model

# In[ ]:

sim = nengo.Simulator(model)
sim.run(3.)


# ## Plot the results

# In[ ]:

plt.figure(figsize=(10, 10))
vocab = model.get_default_vocab(dimensions)

plt.subplot(5, 1, 1)
plt.plot(sim.trange(), model.similarity(sim.data, color_in))
plt.legend(model.get_output_vocab('color_in').keys, fontsize='x-small')
plt.ylabel("color")

plt.subplot(5, 1, 2)
plt.plot(sim.trange(), model.similarity(sim.data, shape_in))
plt.legend(model.get_output_vocab('shape_in').keys, fontsize='x-small')
plt.ylabel("shape")

plt.subplot(5, 1, 3)
plt.plot(sim.trange(), model.similarity(sim.data, cue))
plt.legend(model.get_output_vocab('cue').keys, fontsize='x-small')
plt.ylabel("cue")

plt.subplot(5, 1, 4)
for pointer in ['RED * CIRCLE', 'BLUE * SQUARE']:
    plt.plot(sim.trange(), vocab.parse(pointer).dot(sim.data[conv].T), label=pointer)
plt.legend(fontsize='x-small')
plt.ylabel("convolved")

plt.subplot(5, 1, 5)
plt.plot(sim.trange(), spa.similarity(sim.data[out], vocab))
plt.legend(model.get_output_vocab('out').keys, fontsize='x-small')
plt.ylabel("Output")
plt.show()