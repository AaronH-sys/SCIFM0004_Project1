"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""

import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

cimport cython
cimport numpy as np

from cython.parallel cimport prange
cimport openmp

from libc.math cimport cos, exp, sin, sqrt

ctypedef np.int32_t int_t
ctypedef np.float64_t real_t

#=======================================================================
def initdat(Py_ssize_t nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    return np.random.random_sample((nmax,nmax))*2.0*np.pi
#=======================================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================
def one_energy_wrapper(real_t[:, ::1] arr, Py_ssize_t ix, Py_ssize_t iy, Py_ssize_t nmax):
    return one_energy(arr, ix, iy, nmax)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline real_t one_energy(real_t[:, ::1] arr, Py_ssize_t ix, Py_ssize_t iy, Py_ssize_t nmax) noexcept nogil:
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """

    cdef:
        real_t en, ang

        Py_ssize_t ixp, ixm, iyp, iym

    en = 0.0

    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1+nmax)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1+nmax)%nmax #    

    #
    # Add together the 4 neighbour contributions
    # to the energy
    #
    ang = arr[ix,iy]-arr[ixp,iy]
    en += cos(ang)*cos(ang)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += cos(ang)*cos(ang)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += cos(ang)*cos(ang)
    ang = arr[ix,iy]-arr[ix,iym]
    en += cos(ang)*cos(ang)

    # 4 = 4 neighbours
    return 0.5*(4-3*en)
#=======================================================================
def all_energy_wrapper(real_t[:, ::1] arr, Py_ssize_t nmax, int_t threads):
    return all_energy(arr, nmax, threads)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef real_t all_energy(real_t [:, ::1] arr, Py_ssize_t nmax, int_t threads) noexcept nogil:
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """
    cdef:
        real_t enall

        Py_ssize_t i, j

    enall = 0.0

    for i in prange(nmax, num_threads=threads):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall
#=======================================================================
def get_order_wrapper(real_t[:, ::1] arr, Py_ssize_t nmax, int_t threads):
    return get_order(arr, nmax, threads)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef real_t get_order(real_t[:, ::1] arr, Py_ssize_t nmax, int_t threads) noexcept nogil:
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    cdef:
        real_t Qxx, Qxy, Qyy

        Py_ssize_t i, j

        real_t c, s

        real_t factor

    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #

    Qxx = 0.0
    Qxy = 0.0
    Qyy = 0.0

    factor = 3.0/(2.0*nmax*nmax)
    
    for i in prange(nmax, num_threads=threads):
        for j in range(nmax):
            c = cos(arr[i,j])
            s = sin(arr[i,j])

            Qxx += c*c
            Qxy += c*s
            Qyy += s*s

    Qxx = Qxx*factor - 0.5
    Qxy = Qxy*factor
    Qyy = Qyy*factor - 0.5

    return 0.25 + 0.5*(sqrt((Qxx-Qyy)*(Qxx-Qyy) + 4*(Qxy)*(Qxy)))
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def MC_step(real_t[:, ::1] arr, real_t Ts, Py_ssize_t nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.

    cdef:
        real_t scale, accept

        int_t[:, ::1] xran = np.zeros((nmax,nmax), dtype=np.int32)
        int_t[:, ::1] yran = np.zeros((nmax,nmax), dtype=np.int32)
        real_t[:, ::1] aran = np.zeros((nmax,nmax), dtype=np.float64)
        real_t[:, ::1] bran = np.zeros((nmax,nmax), dtype=np.float64)

        Py_ssize_t i, j, ix, iy

        real_t ang, en0, en1, boltz


    scale = 0.1+Ts
    accept = 0

    xran = np.random.randint(0,high=nmax, size=(nmax,nmax), dtype=np.int32)
    yran = np.random.randint(0,high=nmax, size=(nmax,nmax), dtype=np.int32)
    aran = np.random.normal(scale=scale, size=(nmax,nmax))
    bran = np.random.uniform(0.0,1.0, size=(nmax,nmax))

    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i,j]
            iy = yran[i,j]
            ang = aran[i,j]
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                accept += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = exp( -(en1 - en0) / Ts )

                if boltz >= bran[i,j]:
                    accept += 1
                else:
                    arr[ix,iy] -= ang
    return accept/(nmax*nmax)
#=======================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def main(program, Py_ssize_t nsteps, Py_ssize_t nmax, real_t temp, pflag, int_t threads):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """

    cdef:
        # Create and initialise lattice
        real_t[:, ::1] lattice = np.zeros((nmax,nmax), dtype=np.float64)

        # Create arrays to store energy, acceptance ratio and order parameter
        real_t[::1] energy = np.zeros(nsteps+1,dtype=np.float64)
        real_t[::1] ratio = np.zeros(nsteps+1,dtype=np.float64)
        real_t[::1] order = np.zeros(nsteps+1,dtype=np.float64)

        Py_ssize_t it

        real_t initial, final, runtime

    
    lattice = initdat(nmax)

    # Plot initial frame of lattice
    plotdat(lattice,pflag,nmax)
    
    # Set initial values in arrays
    energy[0] = all_energy(lattice,nmax,threads)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax,threads)

    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax)
        energy[it] = all_energy(lattice,nmax,threads)
        order[it] = get_order(lattice,nmax,threads)
    final = time.time()
    runtime = final-initial
    
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    # Plot final frame of lattice and generate output file
    savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(lattice,pflag,nmax)

    return [nsteps, nmax, temp, order, runtime]
#=======================================================================
