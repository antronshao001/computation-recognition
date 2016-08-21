import numpy as np
import matplotlib.pyplot as plt

import nengo
import nengo_gui
from nengo.processes import WhiteNoise


# ## Step 1: Create the model
#
# Like previous examples, the network consists of `pre`, `post`, and `error` ensembles.
# We'll use two-dimensional white noise input and attempt to learn the product
# using the actual product to compute the error signal.

# In[ ]:

model = nengo.Network()
with model:
    # -- input and pre popluation
    inp = nengo.Node(WhiteNoise(60, high=5).f(d=2))#create 2d white noise
    pre = nengo.Ensemble(120, dimensions=2)#2d white noise in
    nengo.Connection(inp, pre)#connect 2d white noise

    # -- error population
    prod_node = nengo.Node(lambda t, x: x[0] * x[1], size_in=2)  # We'll give it the actual product
    nengo.Connection(inp, prod_node, synapse=None)
    error = nengo.Ensemble(60, dimensions=1)
    nengo.Connection(prod_node, error)

    # -- inhibit error after 40 seconds
    inhib = nengo.Node(lambda t: 2.0 if t > 40.0 else 0.0)
    nengo.Connection(inhib, error.neurons, transform=[[-1]] * error.n_neurons)

    # -- post population
    post = nengo.Ensemble(60, dimensions=1)
    nengo.Connection(post, error, transform=-1)
    error_conn = nengo.Connection(error, post, modulatory=True)
    nengo.Connection(pre, post,
                     function=lambda x: np.random.random(1),
                     learning_rule_type=nengo.PES(error_conn))

    # -- probes
    prod_p = nengo.Probe(prod_node)
    pre_p = nengo.Probe(pre, synapse=0.01)
    post_p = nengo.Probe(post, synapse=0.01)
    error_p = nengo.Probe(error, synapse=0.03)

sim = nengo.Simulator(model)
sim.run(60)
#nengo_gui.Viz(__file__).start(port=8083)


# In[ ]:

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(sim.trange(), sim.data[pre_p], c='b')
plt.legend(('Pre decoding',), loc='best')
plt.subplot(3, 1, 2)
plt.plot(sim.trange(), sim.data[prod_p], c='k', label='Actual product')
plt.plot(sim.trange(), sim.data[post_p], c='r', label='Post decoding')
plt.legend(loc='best')
plt.subplot(3, 1, 3)
plt.plot(sim.trange(), sim.data[error_p], c='b')
plt.ylim(-1, 1)
plt.legend(("Error",), loc='best');


# Let's zoom in on the network at the beginning.

# In[ ]:

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(sim.trange()[:2000], sim.data[pre_p][:2000], c='b')
plt.legend(('Pre decoding',), loc='best')
plt.subplot(3, 1, 2)
plt.plot(sim.trange()[:2000], sim.data[prod_p][:2000], c='k', label='Actual product')
plt.plot(sim.trange()[:2000], sim.data[post_p][:2000], c='r', label='Post decoding')
plt.legend(loc='best')
plt.subplot(3, 1, 3)
plt.plot(sim.trange()[:2000], sim.data[error_p][:2000], c='b')
plt.ylim(-1, 1)
plt.legend(("Error",), loc='best');


# And now right where we turn off learning.

# In[ ]:

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(sim.trange()[38000:42000], sim.data[pre_p][38000:42000], c='b')
plt.legend(('Pre decoding',), loc='best')
plt.subplot(3, 1, 2)
plt.plot(sim.trange()[38000:42000], sim.data[prod_p][38000:42000], c='k', label='Actual product')
plt.plot(sim.trange()[38000:42000], sim.data[post_p][38000:42000], c='r', label='Post decoding')
plt.legend(loc='best')
plt.subplot(3, 1, 3)
plt.plot(sim.trange()[38000:42000], sim.data[error_p][38000:42000], c='b')
plt.ylim(-1, 1)
plt.legend(("Error",), loc='best');
plt.show()

# You can see that it has learned a decent approximation of the product,
# but it's not perfect -- typically, it's not as good as the offline optimization.
# The reason for this is that we've given it white noise input,
# which has a mean of 0; since this happens in both dimensions,
# we'll see a lot of examples of inputs and outputs near 0.
# In other words, we've oversampled a certain part of the
# vector space, and overlearned decoders that do well in
# that part of the space. If we want to do better in other
# parts of the space, we would need to construct an input
# signal that evenly samples the space.