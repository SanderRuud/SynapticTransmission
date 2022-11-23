
import skfdiff
from skfdiff import Model
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

from pylab import *

from main import sim1d

def get_N_init():
    field_vals, t_vals, end_val = sim1d(N_val=1e4, n_xvals=1000, a=1000)
    N_init = field_vals[-1].data_vars["N"].to_numpy()
    N_init = N_init + field_vals[-1].data_vars["C"].to_numpy()
    return N_init

def clearance_of_neurotransmitters1d():

    N_init = get_N_init()

    bc = defaultdict(lambda: ("dirichlet"))
    model = Model(["a*dxxN-k3*N*T+k4*TN",
                    "-k3*N*T+k4*TN+k5*TN",
                    "k3*N*T-k4*TN-k5*TN",
                    "k5*TN"
                ],
                ["N(x)", "T(x)", "TN(x)", "NIN(x)"], ["k3", "k4","k5", "a"],
                boundary_conditions=bc)

    x = np.linspace(0, 44, 1000, endpoint=False) #0 to 1e^-4
    N = N_init
    TN = np.zeros(shape = x.shape)
    NIN = np.zeros(shape = x.shape)

    T = np.zeros(x.shape)
    T[0:5] = 3
    T[-5:] = 3
    k3 = 4e6
    k4 = 1
    k5 = 4e4
    a = 8e1 #8×1e-7

    fields = model.fields_template(x=x, N=N, T=T, TN=TN, NIN=NIN, k3=k3, k4=k4, k5=k5, a=a)
    simulation = skfdiff.Simulation(model, fields, dt=.05, tmax=3)
    container = simulation.attach_container()
    tmax, final_fields = simulation.run()
    plt.plot(container.data.N.t, (container.data.mean(['x']).N).to_numpy(), "r-", label = r"$c_{N}$")
    plt.plot(container.data.T.t, (container.data.mean(['x']).T).to_numpy(), "r--", label = r"$c_{T}$")
    plt.plot(container.data.TN.t, (container.data.mean(['x']).TN).to_numpy(), "b-", label = r"$c_{T-N}$")
    plt.plot(container.data.NIN.t, (container.data.mean(['x']).NIN).to_numpy(), "b--", label = r"$c_{N_{inactive}}$")
    plt.legend()
    plt.title(f"$k_2$ = {k3}, $k_{-2}$ = {k4}, $k_3$ = {k5}, $a$ = {a}")
    plt.ylabel("Avg. concentration")
    plt.xlabel("$t$")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    clearance_of_neurotransmitters1d()
