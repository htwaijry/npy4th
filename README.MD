#npy4th

A package to load/save numpy files for torch7.

Reads .npy and .npz files, and stores torch.*Tensor in .npy format.

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

-- save a torch.*Tensor into a .npy file
myTensor = torch.Tensor({1,2,3})
npy4th.savenpy('tensor.npy', myTensor)
```
