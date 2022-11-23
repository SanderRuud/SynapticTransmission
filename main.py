import skfdiff
from skfdiff import Model
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from pylab import *
import xarray

def inner_elem(a, value, size):
    n = a.shape[0]
    r = np.minimum(np.arange(n)[::-1], np.arange(n))
    inner = np.minimum(r[:,None],r)>size
    a[inner] = value

def inner_elem3D(U, value, size): 
    for i in range(3):
        inner_elem(U[:,:,i], value, size)

def plot_stuff(container, X, Y, Z, dim="3d", t_indices = [0, 5, 10, 30], opacity=1):
    for f in range(3):
        if f == 0:
            cunt = container.data_vars["R"]
            letter = "R"
        elif f == 1:
            cunt = container.data_vars["N"]
            letter = "N"
        else:
            cunt = container.data_vars["C"]
            letter = "R-N"    
        print(letter)
        fig = plt.figure(figsize=(13, 15))
        i = 0
        for t in t_indices:
            if dim =="1d":
                data = cunt.to_numpy()[t,:]
            elif dim == "3d":
                data = cunt.to_numpy()[t,:,:,:]
            else:
                data = cunt.to_numpy()[t,:,:]
            ax = fig.add_subplot(1, len(t_indices), i+1, projection='3d')

            p = ax.scatter(X, Y, Z, c=data, lw=0, s=10, alpha=opacity)
            ax.set_title("$C_{"+letter+"}$, at timestep="+str(t))
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.view_init(20,20)
            i=i+1
                
        fig.colorbar(p, fraction = 0.025, norm=matplotlib.colors.LogNorm())
        plt.show()

def conc_plot(field_vals, t_vals, k1, k2, a):
    R = [np.average(x.data_vars["R"].values) for x in field_vals]
    N = [np.average(x.data_vars["N"].values) for x in field_vals]
    C = [np.average(x.data_vars["C"].values) for x in field_vals]
    plt.plot(t_vals, R, "g.", label = "c_{R}")
    plt.plot(t_vals, N, "r-", label = "c_{N}")
    plt.plot(t_vals, C, "b", label = "c_{R-N}")
    plt.legend()
    plt.title(f"k1 = {k1},  k2 = {k2}, a = {a}")
    plt.ylabel("Avg. concentration")
    plt.xlabel("t")
    plt.show()    

def sim1d(geometry="square", tmax=50*1e-6, dt=.05, unif_N=True, k1=5e6, k2=5, a = 8e5, N_val=1, n_xvals=20, plot_conc=True, convection=False):
    """
    geometry: square/thin
    """
    x = np.linspace(0, 0.44, n_xvals, endpoint = False)
    ε = 0.3
    R = np.zeros(shape = len(x))
    if geometry == "square":
        R[:int(ε*len(x)/2)] = 1 / len(x)
        int(ε*len(x)/2)
    else:
        
        ε_mask = np.arange(int(.5*len(x)*(1-ε)), int(.5*len(x)*(1+ε)))
        R[ε_mask] = 1
    bc = defaultdict(lambda: ("dirichlet"))

    if convection:
        model = Model(["a*dxxR-k1*N*R+k2*C",
                        "-k1*N*R+k2*C+100",
                        "k1*N*R-k2*C"
                    ],
                    ["R(x)", "N(x)", "C(x)"], ["k1", "k2", "a"],
                    boundary_conditions=bc)

    else:

        model = Model(["a*dxxR-k1*N*R+k2*C",
                    "-k1*N*R+k2*C",
                    "k1*N*R-k2*C"
                ],
                ["R(x)", "N(x)", "C(x)"], ["k1", "k2", "a"],
                boundary_conditions=bc)
    
    N = np.zeros(shape = x.shape)
    if not convection:
        N = N + N_val/len(x)
    C = np.zeros(shape = x.shape)
    
    fields = model.fields_template(x=x, R=R, N=N, C=C, k1=k1, k2=k2, a=a)
    simulation = skfdiff.Simulation(model, fields, dt=dt, tmax=tmax)
    t_vals = [0]
    R_0= np.average(R)
    R_current = np.inf
    R_last = -np.inf
    field_vals = []
    field_vals.append(fields)
    last_val = np.nan
    for t, fields in simulation:
        print(t)
        t_vals.append(t)
        field_vals.append(fields)
        if np.allclose(R_current, R_last, rtol=dt): 
            print("No improvement")
            print(t, "in", int(t/dt), "steps")
            break
        R_last = R_current
        R_current = np.average(fields.data_vars["R"])
        if R_current <= 0.5 * R_0:
            print("solution found")
            print(t, "in", int(t/dt), "steps")
            last_val = t_vals[-1]
            break
    if plot_conc:
        conc_plot(field_vals, t_vals, k1, k2, a)    
    return field_vals, t_vals, last_val



def sim2d(geometry="square", tmax=2., dt=.05, unif_N=True, k1=5e6, k2=5, a = 8e5, N_val=1, n_xvals=20, n_yvals = 20, plot_conc=True):
    """
    geometry: square/thin
    """
    x = np.linspace(0, 0.44, n_xvals, endpoint = False)
    y = np.linspace(0, 0.44, n_yvals, endpoint = False)
    R = np.zeros(shape = (len(x), len(y)))
    if geometry == "square":
        inner_elem(R, 1, int(len(x)/3))
    else:
        ε = 0.3
        ε_mask = np.arange(int(.5*len(x)*(1-ε)), int(.5*len(x)*(1+ε)))        
        R[ε_mask] = 1 / len(x) / len(y)

    bc = defaultdict(lambda: ("dirichlet"))
    model = Model(["a*(dxxR+dyyR)-k1*N*R+k2*C",
                    "-k1*N*R+k2*C",
                    "k1*N*R-k2*C"
                ],
                ["R(x, y)", "N(x, y)", "C(x, y)"], ["k1", "k2", "a"],
                boundary_conditions=bc)
    N  = np.zeros(shape = (len(x), len(y))) + (N_val/len(x)/len(y))
    C = np.zeros(shape = (len(x), len(y)))
    fields = model.fields_template(x=x, y=y, R=R, N=N, C=C, k1=k1, k2=k2, a=a)
    simulation = skfdiff.Simulation(model, fields, dt=dt, tmax=tmax)
    t_vals = [0]
    R_0 = np.average(R)

    R_current = np.inf
    R_last = -np.inf
    field_vals = []
    field_vals.append(fields)
    last_val = np.nan
    for t, fields in simulation:
        t_vals.append(t)
        field_vals.append(fields)
        if np.allclose(R_current, R_last, rtol=dt): 
            print("No improvement")
            print(t, "in", int(t/dt), "steps")
            break
        R_last = R_current
        R_current = np.average(fields.data_vars["R"])
        if R_current <= 0.5 * R_0:
            print("solution found")
            print(t, "in", int(t/dt), "steps")
            last_val = t_vals[-1]
            break
    if plot_conc:
        conc_plot(field_vals, t_vals, k1, k2, a)    
    return field_vals, t_vals, last_val



def sim3d(geometry="square", tmax=2., dt=.05, unif_N=True, k1=5e6, k2=5, a = 8e5, N_val=1, n_xvals=20, n_yvals = 20, n_zvals =5, plot_conc=True):
    """
    geometry: square/thin
    """
    x = np.linspace(0, 0.44, n_xvals, endpoint = False)
    y = np.linspace(0, 0.44, n_yvals, endpoint = False)
    z = np.linspace(0, 15e-3, n_zvals)
    R = np.zeros(shape = (len(x), len(y), len(z)))
    if geometry == "square":
        inner_elem3D(R, 1, int(len(x)/3))
    else:
        ε = 0.3
        ε_mask = np.arange(int(.5*len(x)*(1-ε)), int(.5*len(x)*(1+ε)))        
        R[ε_mask] = 1 

    bc = defaultdict(lambda: ("dirichlet"))

    N  = np.zeros(shape = (len(x), len(y), len(z))) + N_val

    
    model = Model(["a*(dxxR+dyyR+dzzR)-k1*N*R+k2*C",
                    "-k1*N*R+k2*C",
                    "k1*N*R-k2*C"
                ],
                ["R(x, y, z)", "N(x, y, z)", "C(x, y, z)"], ["k1", "k2", "a"],
                boundary_conditions=bc)

    C = np.zeros(shape = (len(x), len(y), len(z)))
    fields = model.fields_template(x=x, y=y, z=z, R=R, N=N, C=C, k1=k1, k2=k2, a=a)
    simulation = skfdiff.Simulation(model, fields, dt=dt, tmax=tmax)
    t_vals = [0]
    R_0 = np.average(R)
    R_current = np.inf
    R_last = -np.inf
    field_vals = []
    field_vals.append(fields)
    last_val = np.nan
    for t, fields in simulation:
    
        t_vals.append(t)
        field_vals.append(fields)
        if np.allclose(R_current, R_last, rtol=dt): 
            print("No improvement")
            print(t, "in", int(t/dt), "steps")
            break
        R_last = R_current
        R_current = np.average(fields.data_vars["R"])
        if R_current <= 0.5 * R_0:
            print("solution found")
            print(t, "in", int(t/dt), "steps")
            last_val = t_vals[-1]
            break
    if plot_conc:
        conc_plot(field_vals, t_vals, k1, k2, a)    
    return field_vals, t_vals, last_val

def get_times_until_transmission():
    end_times = []
    N_vals = np.logspace(10, -5, 24)
    for N_val in N_vals:
        print(N_val)
        field_vals, t_vals, end_val = sim1d(N_val=N_val, dt=1e-6, tmax=1000*(1e-6), plot_conc=False)
        if np.isnan(end_val):
            end_times = end_times + [np.nan]*(len(N_vals)-len(end_times))
            break
        end_times.append(end_val)
    print(end_times)
    N_needed = N_vals[np.nanargmax(end_times)]
    print(N_needed)
    plt.loglog(end_times, N_vals, "r.")
    plt.xlabel("Initial concentration of neurotransmitters")
    plt.ylabel("$t_{\mathrm{end}}$")
    plt.title("Time until signal. N needed:" + str(round(N_needed, 4)))
    plt.show()

def get1d_plots():
    field_vals, t_vals, end_val = sim1d(N_val=0.1, n_xvals=100, dt=1e-6)
    print(end_val)
    N_init = field_vals[-1].data_vars["N"].to_numpy()
    container = xarray.concat(field_vals, dim="example")
    x = field_vals[0].x
    y = np.array([0])
    z = np.array([0])
    X, Y, Z = np.meshgrid(x, y, z)
    plot_stuff(container, X, Y, Z, dim = "1d", t_indices = [0, 1, 50, 200])

def get1d_convction_plots():
    field_vals, t_vals, end_val = sim1d(N_val=0.1, n_xvals=100, dt=1e-6, convection=True)
    N_init = field_vals[-1].data_vars["N"].to_numpy()
    container = xarray.concat(field_vals, dim="example")
    x = field_vals[0].x
    y = np.array([0])
    z = np.array([0])
    X, Y, Z = np.meshgrid(x, y, z)
    plot_stuff(container, X, Y, Z, dim = "1d", t_indices = [0, 1, 49])
    

def get2d_plots():
    field_vals, t_vals, end_val = sim2d(N_val=22, n_xvals=20, n_yvals=20, dt=1e-6, tmax = (1e-6)*50)
    N_init = field_vals[-1].data_vars["N"].to_numpy()
    container = xarray.concat(field_vals, dim="example")
    x = field_vals[0].x
    y = field_vals[0].y
    z = np.array([0])
    X, Y, Z = np.meshgrid(x, y, z)
    plot_stuff(container, X, Y, Z, dim = "2d", t_indices = [0, 1, 3, 4])

def get3d_plots():
    field_vals, t_vals, end_val = sim3d(N_val=5e-2, n_xvals=20, n_yvals=20, n_zvals=5, dt=1e-6, tmax = (1e-6)*10)
    N_init = field_vals[-1].data_vars["N"].to_numpy()
    container = xarray.concat(field_vals, dim="example")
    x = field_vals[0].x
    y = field_vals[0].y
    z = field_vals[0].z
    X, Y, Z = np.meshgrid(x, y, z)
    plot_stuff(container, X, Y, Z, dim = "2d", t_indices = [0, 1, 3, 4])

if __name__ == "__main__":
    #get1d_plots()
    #get2d_plots()
    #get3d_plots()
    #get1d_convction_plots()
    #get_times_until_transmission()
    get1d_plots()


