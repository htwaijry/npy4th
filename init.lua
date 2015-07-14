--Copyright (C) 2015  Hani Altwaijry
--Released under MIT License
--license available in LICENSE file

require 'torch'
require 'libnpy4th'
require 'xlua'

local npy4th = {}

local help = {
loadnpy = [[Loads a numpy .npy file to a torch.Tensor]],
loadnpz = [[Loads a numpy .npz file to a table]]
}

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



return npy4th