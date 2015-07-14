package = "npy4th"
version = "1.0-0"

source = {
   url = "https://github.com/htwaijry/npy4th",
   tag = "1.0-0"
}

description = {
   summary = "An interface between Numpy and Torch",
   detailed = [[
      Load .npy files into torch.
   ]],
   homepage = "https://github.com/htwaijry/npy4th",
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