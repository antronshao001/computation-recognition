__author__ = 'anthony'
import matplotlib.pyplot as plt
import nengo
import nengo_gui
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from nengo.dists import Choice
from nengo.utils.ensemble import tuning_curves

sample = []
for x in np.arange(-1, 1, 0.1):
    for y in np.arange(-1, 1, 0.1):
        sample.append([x, y])

model = nengo.Network(label='2D arm representation')

with model:
    ens_2d = nengo.Ensemble(100, dimensions=2, encoders=Choice(sample))

sim = nengo.Simulator(model)

##### single tuning curve #####
# inputs = np.zeros((50,2))
# inputs[:,0] = np.arange(-1,1,0.04)
# inputs[:,1] = 0.5
# eval_points, activities = tuning_curves(ens_2d, sim, inputs=inputs)
# plt.figure()
# ax = plt.subplot(111)
# ax.plot(inputs.T[0], activities)
# ax.set_ylabel("Firing rate (Hz)")
# ax.set_xlabel("represented value(x)");
# ##### single tuning curve #####

##### 2d plot #####
# inputs = np.zeros((100, 2))
# theta = np.zeros((100,1))
# for i in range(100):
#     theta[i] = (i-50)*np.pi/50
#     inputs[i, 0] = np.cos(theta[i])
#     inputs[i, 1] = np.sin(theta[i])
#
# eval_points, activities = tuning_curves(ens_2d, sim, inputs=inputs)
#
#
# plt.figure(figsize=(10,5))
# ax = plt.subplot(121)
# ax.plot(theta, activities)
# ax.set_ylabel("Firing rate (Hz)")
# ax.set_xlabel("represented angle(move=1)");
#
# eval_points_c, activities_c = tuning_curves(ens_2d, sim, inputs=inputs*0.5)
# ax_c = plt.subplot(122)
# ax_c.plot(theta, activities_c)
# ax_c.set_ylabel("Firing rate (Hz)")
# ax_c.set_xlabel("represented angle(move=0.5)");
##### 2d plot #####

##### 3d plot #####
eval_points, activities = tuning_curves(ens_2d, sim)

plt.figure(figsize=(10, 5))
ax = plt.subplot(121, projection='3d')
ax.set_title("Tuning curve of all neurons")
for i in range(ens_2d.n_neurons):
    ax.plot_surface(eval_points.T[0], eval_points.T[1], activities.T[i], cmap=plt.cm.autumn)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("Firing rate (Hz)");
axo = plt.subplot(122, projection='3d')
axo.set_title("Neuron %d" % 10)
axo.plot_surface(eval_points.T[0], eval_points.T[1], activities.T[10], cmap=plt.cm.autumn)
axo.set_xlabel("$x_1$")
axo.set_ylabel("$x_2$")
axo.set_zlabel("Firing rate (Hz)");
##### 3d plot #####


# nengo_gui.Viz(__file__).start(port=8091)
plt.show()

