import numpy as np
import nengo
import nengo_gui
from nengo.dists import Uniform

model = nengo.Network(label='Two Neurons')
with model:
    neurons = nengo.Ensemble(100, dimensions=2)

import numpy as np
with model:
    sin = nengo.Node(output=np.sin)
    cos = nengo.Node(output=np.cos)

with model:
    # The indices in neurons define which dimension
    # the input will project to
    nengo.Connection(sin, neurons[0])
    nengo.Connection(cos, neurons[1])



nengo_gui.Viz(__file__).start(port=8081)
