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

static void load_array_to_lua(lua_State *L, cnpy::NpyArray& arr){
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

	void * tensorDataPtr = NULL;
	size_t numBytes = 0;

	if ( arr.arrayType == 'f' ){ // float32/64
		if ( arr.word_size == 4 ){ //float32
			THFloatTensor *tensor = THFloatTensor_newWithSize(size, stride);
		    tensorDataPtr = (void *)(THFloatTensor_data(tensor));
		    numBytes = THFloatTensor_nElement(tensor) * arr.word_size;
		    luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.FloatTensor"));
    
		}else if ( arr.word_size ==  8){ //float 64
			THDoubleTensor *tensor = THDoubleTensor_newWithSize(size, stride);
			tensorDataPtr = (void *)(THDoubleTensor_data(tensor));
		    numBytes = THDoubleTensor_nElement(tensor) * arr.word_size;
		    luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.DoubleTensor"));
		}
	}else if ( arr.arrayType == 'i' || arr.arrayType == 'u' ){ // does torch have unsigned types .. need to look
		if ( arr.word_size == 1 ){ //int8
			THByteTensor *tensor = THByteTensor_newWithSize(size, stride);
			tensorDataPtr = (void *)(THByteTensor_data(tensor));
		    numBytes = THByteTensor_nElement(tensor) * arr.word_size;
		    luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.ByteTensor"));
    
		}else if ( arr.word_size == 2 ){ //int16
			THShortTensor *tensor = THShortTensor_newWithSize(size, stride);
			tensorDataPtr = (void *)(THShortTensor_data(tensor));
		    numBytes = THShortTensor_nElement(tensor) * arr.word_size;
		    luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.ShortTensor"));
    
		}else if ( arr.word_size == 4 ){ //int32
			THIntTensor *tensor = THIntTensor_newWithSize(size, stride);
			tensorDataPtr = (void *)(THIntTensor_data(tensor));
		    numBytes = THIntTensor_nElement(tensor) * arr.word_size;
		    luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.IntTensor"));
    
		}else if ( arr.word_size ==  8){ //long 64
			THLongTensor *tensor = THLongTensor_newWithSize(size, stride);
			tensorDataPtr = (void *)(THLongTensor_data(tensor));
		    numBytes = THLongTensor_nElement(tensor) * arr.word_size;
		    luaT_pushudata(L, tensor, luaT_checktypename2id(L, "torch.LongTensor"));
		}
	}else{
		printf("array type unsupported");
		throw std::runtime_error("unsupported data type");
	}

		// now copy the data
		assert(tensorDataPtr);
		memcpy(tensorDataPtr, (void *)(arr.data<void>()), numBytes);


}

static int loadnpy_l(lua_State *L) {

	try{
		const char *filename = lua_tostring(L, 1);

		std::string fpath = std::string(filename);

		cnpy::NpyArray arr = cnpy::npy_load(fpath);

		load_array_to_lua(L, arr);

	} catch (std::exception& e){
		THError(e.what());
	}
	return 1;
}

static int loadnpz_l(lua_State *L){
	try{
		const char *filename = lua_tostring(L, 1);

		std::string fpath = std::string(filename);

		cnpy::npz_t npzData = cnpy::npz_load(filename);

		// create a new table
		lua_newtable(L);  
		int tbl = lua_gettop(L);

		for (cnpy::npz_t::iterator i=npzData.begin(); i!=npzData.end(); ++i){
			std::string name = i->first;
			cnpy::NpyArray arr = i->second;

			lua_pushstring(L, name.c_str());
			load_array_to_lua(L, arr);
			lua_rawset(L, tbl);
		}


	}catch (std::exception& e){
		THError(e.what());
	}

	return 1;
}

static std::vector<size_t> get_shape(int nDimension, long* size){
	std::vector<size_t> shape(nDimension);
	for (int k=0; k< nDimension; k++){
		shape[k] = (size_t) size[k];
	}		

	return shape;
}

static int savenpy_l(lua_State *L){
	try{

		const char *filename = lua_tostring(L, 1);
		std::string fpath = std::string(filename);

		const char *inType = luaT_typename(L, 2);

		int typeId = luaL_checkinteger(L, 3);

        const char *mode = lua_tostring(L, 4);

		bool success=false;
		switch ( typeId ){
		case 0: //double
		{	THDoubleTensor * tensor = (THDoubleTensor *) luaT_checkudata(L, 2, inType);
			std::vector<size_t> shape = get_shape(tensor->nDimension, tensor->size);
			success=cnpy::npy_save<double>(fpath, tensor->storage->data, shape, mode);
		} break;
	
		case 1: //float
		{	THFloatTensor * tensor = (THFloatTensor *) luaT_checkudata(L, 2, inType);
			std::vector<size_t> shape = get_shape(tensor->nDimension, tensor->size);
			success=cnpy::npy_save<float>(fpath, tensor->storage->data, shape, mode);
		} 
		break;
	
		case 2: //int
		{	THIntTensor * tensor = (THIntTensor *) luaT_checkudata(L, 2, inType);
			std::vector<size_t> shape = get_shape(tensor->nDimension, tensor->size);
			success=cnpy::npy_save<int>(fpath, tensor->storage->data, shape, mode);
		} 
		break;


		case 3: //byte
		{	THByteTensor * tensor = (THByteTensor *) luaT_checkudata(L, 2, inType);
			std::vector<size_t> shape = get_shape(tensor->nDimension, tensor->size);
			success=cnpy::npy_save<unsigned char>(fpath, tensor->storage->data, shape, mode);
		} 
		break;

		case 4: //long
		{	THLongTensor * tensor = (THLongTensor *) luaT_checkudata(L, 2, inType);
			std::vector<size_t> shape = get_shape(tensor->nDimension, tensor->size);
			success=cnpy::npy_save<long>(fpath, tensor->storage->data, shape, mode);
		} 
		break;

		case 5: //short
		{	THShortTensor * tensor = (THShortTensor *) luaT_checkudata(L, 2, inType);
			std::vector<size_t> shape = get_shape(tensor->nDimension, tensor->size);
			success=cnpy::npy_save<short>(fpath, tensor->storage->data, shape, mode);
		} 
		break;

		default:
			THError("unsupported tensor type :-(");
		}
		if (!success)
			THError("error saving file. are you sure the path is ok?");
		
		lua_pushboolean(L, success); // to return true

	}catch (std::exception& e){
		THError(e.what());
	}
	return 1;
}


static const struct luaL_reg npyth [] = {
  {"loadnpy", loadnpy_l},
  {"loadnpz", loadnpz_l},
  {"savenpy", savenpy_l},
  {NULL, NULL}
};

extern "C" int luaopen_libnpy4th (lua_State *L) {
  luaL_openlib(L, "libnpy4th", npyth, 0);
  return 1;
}
