import pycuda.driver as cuda
import pycuda.autoinit 
import tensorrt as trt
import numpy as np
import torch
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # This logger is required to build an engine

def is_dimension_dynamic(dim):
    return dim is None or dim <= 0

def is_shape_dynamic(shape):
    return any([is_dimension_dynamic(dim) for dim in shape])

def run_trt_engine(context, engine, tensors):

    bindings = [None]*engine.num_bindings
    for name,tensor in tensors['inputs'].items():
        idx = engine.get_binding_index(name)
        bindings[idx] = tensor.data_ptr()
        if engine.is_shape_binding(idx) and is_shape_dynamic(context.get_shape(idx)):
            context.set_shape_input(idx, tensor)
        elif is_shape_dynamic(engine.get_binding_shape(idx)):
            context.set_binding_shape(idx, tensor.shape)

    for name,tensor in tensors['outputs'].items():
        idx = engine.get_binding_index(name)
        bindings[idx] = tensor.data_ptr()

    context.execute_v2(bindings=bindings)

def load_engine(engine_filepath, trt_logger):
    with open(engine_filepath, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def engine_info(engine_filepath):

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    engine = load_engine(engine_filepath, TRT_LOGGER)

    binding_template = r"""
{btype} {{
  name: "{bname}"
  data_type: {dtype}
  dims: {dims}
}}"""
    type_mapping = {"DataType.HALF": "TYPE_FP16",
                    "DataType.FLOAT": "TYPE_FP32",
                    "DataType.INT32": "TYPE_INT32",
                    "DataType.BOOL" : "TYPE_BOOL"}

    print("engine name", engine.name)
    print("has_implicit_batch_dimension", engine.has_implicit_batch_dimension)
    start_dim = 0 if engine.has_implicit_batch_dimension else 1
    print("num_optimization_profiles", engine.num_optimization_profiles)
    print("max_batch_size:", engine.max_batch_size)
    print("device_memory_size:", engine.device_memory_size)
    print("max_workspace_size:", engine.max_workspace_size)
    print("num_layers:", engine.num_layers)

    for i in range(engine.num_bindings):
        btype = "input" if engine.binding_is_input(i) else "output"
        bname = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        bdims = engine.get_binding_shape(i)
        config_values = {
            "btype": btype,
            "bname": bname,
            "dtype": type_mapping[str(dtype)],
            "dims": list(bdims[start_dim:])
        }
        final_binding_str = binding_template.format_map(config_values)
        print(final_binding_str)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtWrapper():
    def __init__(self, trt_path=""):
        self.trt_path = trt_path
        self.engine = self._get_engine(self.trt_path)
        self.context = self.engine.create_execution_context()

    def _get_engine(self,path):
        print(f"Loading engine from {path}")
        with open(path,'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            return engine

    def _build_engine(self):
        raise NotImplementedError

    def allocate_buffers(self, engine, context):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for i,binding in enumerate(engine):
            size = trt.volume(context.get_binding_shape(i))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def unified_mem_mod(self, engine, context):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for i,binding in enumerate(engine):
            size = trt.volume(context.get_binding_shape(i))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                in_mem = cuda.managed_empty(shape=size, dtype=dtype, mem_flags=cuda.mem_attach_flags.GLOBAL)
                bindings.append(int(in_mem.base.get_device_pointer()))
                inputs.append(in_mem)
            else:
                out_mem = cuda.managed_empty(shape=size, dtype=dtype, mem_flags=cuda.mem_attach_flags.GLOBAL)
                bindings.append(int(out_mem.base.get_device_pointer()))
                outputs.append(out_mem)
        cudacontext.context.synchronize()

        return inputs, outputs, bindings, stream

    def run_trt_engine(self,tensors):
        bindings = [None] * self.engine.num_bindings
        for name,tensor in tensors['inputs'].items():
            idx = self.engine.get_binding_index(name)
            bindings[idx] = tensor.data_ptr()
            if self.engine.is_shape_binding(idx) and is_shape_dynamic(self.context.get_shape(idx)):
                self.context.set_shape_input(idx, tensor)
            elif is_shape_dynamic(self.engine.get_binding_shape(idx)):
                self.context.set_binding_shape(idx, tensor.shape)

        for name,tensor in tensors['outputs'].items():
            idx = self.engine.get_binding_index(name)
            bindings[idx] = tensor.data_ptr()

        self.context.execute_v2(bindings=bindings)

    def do_inference(self):
        raise NotImplementedError 

class EncWrapper(TrtWrapper):
    def __init__(self,trt_path):
        super(EncWrapper,self).__init__(trt_path)
    
    def do_inference(self, src, src_mask, batch_size=1):
        with self.engine.create_execution_context() as context:
            context.active_optimization_profile = 0
            shapes = [src.shape, src_mask.shape]
            # reshape bindings before doing inference
            for i,shape in enumerate(shapes):
                binding_shape = context.get_binding_shape(i)
                if -1 in binding_shape:
                    binding_shape = tuple(shape)
                    context.set_binding_shape(i,(binding_shape))
            inputs, outputs, bindings, stream = self.allocate_buffers(self.engine,context)
            inputs[0].host = src
            inputs[1].host = src_mask

            # Transfer data from CPU to the GPU.
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            # Run inference.
            context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            # Synchronize the stream
            stream.synchronize()
            # Return only the host outputs.
            return [out.host for out in outputs]
            
class DecInitWrapper(TrtWrapper):
    def __init__(self,trt_path):
        super(DecInitWrapper,self).__init__(trt_path)

    def do_inference(self, decoder_states, decoder_mask, encoder_states, encoder_mask, batch_size=1):
        with self.engine.create_execution_context() as context:
            context.active_optimization_profile = 0
            shapes = [decoder_states.shape, decoder_mask.shape, encoder_states.shape, encoder_mask.shape]
            # reshape bindings before doing inference
            for i,shape in enumerate(shapes):
                binding_shape = context.get_binding_shape(i)
                if -1 in binding_shape:
                    binding_shape = tuple(shape)
                    context.set_binding_shape(i,(binding_shape))
            inputs, outputs, bindings, stream = self.allocate_buffers(self.engine,context)
            inputs[0].host = decoder_states
            inputs[1].host = decoder_mask
            inputs[2].host = encoder_states
            inputs[3].host = encoder_mask

            # Transfer data from CPU to the GPU.
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            # Run inference.
            context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            # Synchronize the stream
            stream.synchronize()
            # Return only the host outputs.
            return [out.host for out in outputs]

class DecWrapper(TrtWrapper):
    def __init__(self,trt_path):
        super(DecWrapper,self).__init__(trt_path)

    def do_inference(self, decoder_states, decoder_mask, encoder_states, encoder_mask, decoder_mems, batch_size=1):
        with self.engine.create_execution_context() as context:
            context.active_optimization_profile = 0
            shapes = [decoder_states.shape, decoder_mask.shape, encoder_states.shape, encoder_mask.shape, decoder_mems.shape]
            # reshape bindings before doing inference
            for i,shape in enumerate(shapes):
                binding_shape = context.get_binding_shape(i)
                if -1 in binding_shape:
                    binding_shape = tuple(shape)
                    context.set_binding_shape(i,(binding_shape))
            inputs, outputs, bindings, stream = self.allocate_buffers(self.engine,context)
            inputs[0].host = decoder_states
            inputs[1].host = decoder_mask
            inputs[2].host = encoder_states
            inputs[3].host = encoder_mask
            inputs[4].host = decoder_mems

            # Transfer data from CPU to the GPU.
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            # Run inference.
            context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            # Synchronize the stream
            stream.synchronize()
            # Return only the host outputs.
            return [out.host for out in outputs]

