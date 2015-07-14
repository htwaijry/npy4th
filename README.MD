#npy4th

A package to load numpy .npy and .npz files into torch.

### Requirements
* torch7 
* xlua

### Installation
You can install the package as follows:

```
clone https://github.com/htwaijry/npy4th.git
cd npy4th
luarocks make
```

### Use
```
npy4th = require 'npy4th'

-- read a .npy file into a torch tensor
array = npy4th.loadnpy('array.npy')

-- read a .npz file into a table
tbl = npy4th.loadnpz('table.npz')
```