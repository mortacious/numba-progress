# Numba-progress
 
A progress bar implementation for numba functions using tqdm.
The module provides the class `ProgressBar` that works as a wrapper around the 
`tqdm.tqdm` progress bar. 

It works by spawning a separate thread that updates the `tqdm` progress bar 
based on an atomic counter which can be accessed and updated in a numba nopython function.

The progress bar works with parallel as well as sequential numba functions.

## Installation

### Using pip
```
pip install numba-progress
```

### From source
```
git clone https://github.com/mortacious/numba-progress.git
cd numba-progress
python setup.py install
```

## Usage

```python
from numba import njit
from numba_progress import ProgressBar

num_iterations = 100

@njit(nogil=True)
def numba_function(num_iterations, progress_proxy):
    for i in range(num_iterations):
        #<DO CUSTOM WORK HERE>
        progress_proxy.update(1)

with ProgressBar(total=num_iterations) as progress:
    numba_function(num_iterations, progress)
```

The `ProgressBar` also works within parallel functions out of the box:

```python
from numba import njit, prange
from numba_progress import ProgressBar

num_iterations = 100

@njit(nogil=True, parallel=True)
def numba_function(num_iterations, progress_proxy):
    for i in prange(num_iterations):
        #<DO CUSTOM WORK HERE>
        progress_proxy.update(1)

with ProgressBar(total=num_iterations) as progress:
    numba_function(num_iterations, progress)
```

Refer to the `examples` folder for more usage examples.