//Copyright (C) 2015  Hani Altwaijry
//Released under MIT License
//license available in LICENSE file

#include <string>
#include <cstdio>
#include <cstdlib>
#include <cnpy.h>
#include <stdint.h>

#include <TH.h>
#include <luaT.h>


static int loadnpy_l(lua_State *L) {

	try{
	const char *filename = lua_tostring(L, 1);

	std::string fpath = std::string(filename);

	cnpy::NpyArray arr = cnpy::npy_load(fpath);

	int ndims = arr.shape.size();

	//based on code from mattorch with stride fix
	int k;
    THLongStorage *size = THLongStorage_newWithSize(ndims);
    THLongStorage *stride = THLongStorage_newWithSize(ndims);
    for (k=0; k<ndims; k++) {
      THLongStorage_set(size, k, arr.shape[k]);
      if (k > 0)
        THLongStorage_set(stride, ndims-k-1, arr.shape[ndims-k]*THLongStorage_get(stride,ndims-k));
      else
        THLongStorage_set(stride, ndims-k-1, 1);
    }

	if ( arr.arrayType == 'f' ){ // float32/64
		if ( arr.word_size == 4 ){ //float32
			THFloatTensor *tensor = THFloatTensor_newWithSize(size, stride);
		      memcpy((void *)(THFloatTensor_data(tensor)),
		             (void *)(arr.data<void>()), THFloatTensor_nElement(tensor) * sizeof(float));
		      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.FloatTensor"));
    
		}else if ( arr.word_size ==  8){ //float 64
			THDoubleTensor *tensor = THDoubleTensor_newWithSize(size, stride);
		      memcpy((void *)(THDoubleTensor_data(tensor)),
		             (void *)(arr.data<void>()), THDoubleTensor_nElement(tensor) * sizeof(double));
		      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.DoubleTensor"));
		}
	}else if ( arr.arrayType == 'i' || arr.arrayType == 'u' ){ // does torch have unsigned types .. need to look
		if ( arr.word_size == 1 ){ //int8
			THByteTensor *tensor = THByteTensor_newWithSize(size, stride);
		      memcpy((void *)(THByteTensor_data(tensor)),
		             (void *)(arr.data<void>()), THByteTensor_nElement(tensor) * sizeof(int8_t));
		      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.ByteTensor"));
    
		}else if ( arr.word_size == 2 ){ //int16
			THShortTensor *tensor = THShortTensor_newWithSize(size, stride);
		      memcpy((void *)(THShortTensor_data(tensor)),
		             (void *)(arr.data<void>()), THShortTensor_nElement(tensor) * sizeof(int16_t));
		      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.ShortTensor"));
    
		}else if ( arr.word_size == 4 ){ //int32
			THIntTensor *tensor = THIntTensor_newWithSize(size, stride);
		      memcpy((void *)(THIntTensor_data(tensor)),
		             (void *)(arr.data<void>()), THIntTensor_nElement(tensor) * sizeof(int32_t));
		      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.IntTensor"));
    
		}else if ( arr.word_size ==  8){ //long 64
			THLongTensor *tensor = THLongTensor_newWithSize(size, stride);
		      memcpy((void *)(THLongTensor_data(tensor)),
		             (void *)(arr.data<void>()), THLongTensor_nElement(tensor) * sizeof(int64_t));
		      luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.LongTensor"));
		}
	}else{
		throw std::runtime_error("unsupported data type");
	}
	} catch (std::exception& e){
		THError(e.what());
	}
	return 1;
}

static const struct luaL_reg npyth [] = {
  {"loadnpy", loadnpy_l},
  {NULL, NULL}
};

extern "C" int luaopen_libnpy4th (lua_State *L) {
  luaL_openlib(L, "libnpy4th", npyth, 0);
  return 1;
}