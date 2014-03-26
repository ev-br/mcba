from itertools import ifilter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import mcba.db as db
from mcba.models.impurity.helpers import cnfProxy 

from mcba.helpers import arr_from_dictiter

from mcba.models.impurity import SingleImpurity, gamma, initial_q, k_F, E_FS

"""
Pull the simulation from a DB and plot overlaps vs energies
for partitions with large enough of a weight
"""

def _plot_title(par):
    return r"$N = {0}$      $\gamma={1:.2f}$    $q/k_F = {2:.2f}$".format(
            par.N, gamma(par), initial_q(par)/k_F(par))

def ene_hist(ax, data, par):
    """Plot the overlaps with the FS versus energies, color-coding the 
    value of <f_q|P|f_q>, adding the plot to the axis `ax`.
    """
    sc = ax.scatter(data["energy"], data["FSfq"], c = data["P"], s = 75,  
            edgecolor='none', cmap = cm.RdYlBu_r)
    plt.colorbar(sc, use_gridspec=True)

    #fine-tune the plot
    ax.grid(True)
    ax.set_xlim(0, ax.get_xlim()[1])

    title = _plot_title(par)
    ax.set_title(title, fontsize = 22)
    ax.set_ylabel(r'$\log_{10}{ | \langle \mathrm{FS} | f_q \rangle |^2}$', 
            fontsize=22)
    ax.set_xlabel(r'$(E-E_{FS})/E_{FS}$', fontsize=18)
    
    plt.gcf().tight_layout()

    return ax


def distr_P(ax, data, par):
    """Plot <f|P|f> vs energy."""
    sc = ax.scatter(data["energy"], data["P"], s = 25 )
    ax.grid(True)

    title = _plot_title(par)
    ax.set_title(title, fontsize = 22)
    ax.set_ylabel(r'$ | \langle f_q | P | f_q \rangle $', 
            fontsize=22)
    ax.set_xlabel(r'$(E-E_{FS})/E_{FS}$', fontsize=18)

    return ax



if __name__ == "__main__":
    # load the configurations from the DB (run mc.py to produce one)
    db_fname = "N3mc.sqlite"
    handle, = db.get_handles(db_fname)
    par = db.get_param(handle)
    model = SingleImpurity(par)

    # get data
    pred = lambda x: x["FSfq"]**2 >8e-5    
    it = ifilter(pred, db.row_iterator(handle))
    proxy_it = (cnfProxy(item, model) for item in it)

    dt = np.dtype([("fs_pairs", "object"), ("energy", "float64"), 
                   ("FSfq", "float64"), ("P", "float64")])
    data = arr_from_dictiter(proxy_it, dt)
    
    data["energy"] = (data["energy"] - E_FS(par)) / E_FS(par)
    data["FSfq"] = np.log10( data["FSfq"]**2 )

    # produce plots
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = ene_hist(ax, data, par)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1 = distr_P(ax1, data, par)

    plt.show()

