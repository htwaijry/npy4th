package = "npy4th"
version = "1.3-1"

source = {
   url = "git://github.com/htwaijry/npy4th",
   tag = "1.3-1"
}

description = {
   summary = "A Numpy format I/O library for Torch",
   detailed = [[
      Load .npy and .npz files into torch, and save torch.*Tensor into .npy files.
   ]],
   homepage = "git://github.com/htwaijry/npy4th",
   license = "MIT"
}

dependencies = {
   "torch >= 7.0",
   "xlua >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
