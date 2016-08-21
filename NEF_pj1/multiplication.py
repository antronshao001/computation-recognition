import matplotlib.pyplot as plt
import numpy as np
import nengo
import nengo_gui
from nengo.dists import Choice
from nengo.utils.functions import piecewise

# Create the model object
model = nengo.Network(label='Multiplication')
with model:
    # Create 4 ensembles of leaky integrate-and-fire neurons
    A = nengo.Ensemble(500, dimensions=1)
    B = nengo.Ensemble(500, dimensions=1)
    combined = nengo.Ensemble(500, dimensions=2)
    prod = nengo.Ensemble(500, dimensions=1)
    prod_learn = nengo.Ensemble(500, dimensions=1)
    error = nengo.Ensemble(500,dimensions=1)

    combined.encoders = Choice([[1,1],[-1,1],[1,-1],[-1,-1]])

    # Create a piecewise step function for input
    inputA = nengo.Node(nengo.processes.WhiteNoise(50, high=1).f(d=1))
    inputB = nengo.Node(nengo.processes.WhiteNoise(50, high=1).f(d=1))

    # Connect the input nodes to the appropriate ensembles
    nengo.Connection(inputA, A)
    nengo.Connection(inputB, B)

    # Connect input ensembles A and B to the 2D combined ensemble
    nengo.Connection(A, combined[0])
    nengo.Connection(B, combined[1])

    # Connect the combined ensemble to the output ensemble D
    nengo.Connection(combined, prod, function=lambda x: x[0]*x[1])
    nengo.Connection(prod, error)
    nengo.Connection(prod_learn, error, transform=-1)
    error_con = nengo.Connection(error, prod_learn, modulatory=True)
    nengo.Connection(combined, prod_learn, function=lambda x: np.random.random(1),
                     learning_rule_type=nengo.PES(error_con, learning_rate=0.001))

    inputA_probe = nengo.Probe(inputA)
    inputB_probe = nengo.Probe(inputB)
    A_probe = nengo.Probe(A, synapse=0.01)
    B_probe = nengo.Probe(B, synapse=0.01)
    prod_probe = nengo.Probe(prod, synapse=0.01)
    prod_learn_probe = nengo.Probe(prod_learn, synapse=0.01)
    error_probe = nengo.Probe(error, synapse=0.01)

# Create the simulator
sim = nengo.Simulator(model)
#nengo_gui.Viz(__file__).start(port=8086)
# Run it for 5 seconds
sim.run(50)
# Plot the input signals and decoded ensemble values
# plt.plot(sim.trange(), sim.data[A_probe], label="Decoded A")
# plt.plot(sim.trange(), sim.data[B_probe], label="Decoded B")
# plt.plot(sim.trange(), sim.data[prod_learn_probe], label="Decoded product")
# plt.plot(sim.trange(), sim.data[prod_probe], c='k', label="Actual product")
plt.plot(sim.trange(), np.abs(sim.data[error_probe]), c='k', label="Absolute Error")
plt.legend(loc='best')
plt.ylim(-2, 2);
plt.show()