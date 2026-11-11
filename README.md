# SCIFM0004 AC Mini Project

Navigate through different implementations using the tags, the highest version number of each implementation are the ones
used in the report.

## Running Code
### Python, NumPy, Numba
```
python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
```

### Cython
```
python setup_LebwohlLasher_pyx.py build_ext -fi
python run_LebwohlLasher_pyx.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
```
On my M1 Mac I had to run the command below after running the setup script, otherwise the program wouldn't run.
```
install_name_tool -delete_rpath /opt/anaconda3/lib LebwohlLasher_pyx.cpython-312-darwin.so
```

### Parallel Numba Cython
Run the same way as their serial counterparts, with the number of threads being passed after the plot flag 
