--Copyright (C) 2016  Hani Altwaijry
--Released under MIT License
--license available in LICENSE file

require 'torch'
require 'libnpy4th'
require 'xlua'

local npy4th = {}

local help = {
loadnpy = [[loadnpy(filepath) -- Loads a numpy .npy file to a torch.Tensor]],
loadnpz = [[loadnpz(filepath) -- Loads a numpy .npz file to a table]],
savenpy = [[savenpy(filepath, tensor) -- Saves a torch tensor in .npy format]]
}

local typeIds = {}
typeIds['torch.DoubleTensor']=0
typeIds['torch.FloatTensor']=1
typeIds['torch.IntTensor']=2
typeIds['torch.ByteTensor']=3
typeIds['torch.LongTensor']=4
typeIds['torch.ShortTensor']=5
typeIds['torch.CudaTensor']=1 -- saved as float



npy4th.loadnpy = function(filepath)
                   if not filepath then
                      xlua.error('file path must be supplied',
                                  'npy4th.loadnpy', 
                                  help.loadnpy)
                   end
                   return libnpy4th.loadnpy(filepath)
                end

npy4th.loadnpz = function(filepath)
                   if not filepath then
                      xlua.error('file path must be supplied',
                                  'npy4th.loadnpz', 
                                  help.loadnpz)
                   end
                   return libnpy4th.loadnpz(filepath)
                end

npy4th.savenpy = function(filepath, tensor, mode)
		  if not filepath then
			xlua.error('file path must be supplied', 
					'npy4th.savenpy', 
					help.savenpy)
		  end
		  if not tensor or (type(tensor) =='userdata' and tensor.__typename ~= nil and typeIds[tensor:type()] == nil ) then
			xlua.error('Must pass a torch.*Tensor or unsupported tensor type', 'npy4th.savenpy', help.savenpy)
		  end
		  if tensor:type()=='torch.CudaTensor' then
			tensor = tensor:float() -- convert it to float
		  end

          mode = mode or 'w'

		  return libnpy4th.savenpy(filepath, tensor, typeIds[tensor:type()], mode)
		end

return npy4th
