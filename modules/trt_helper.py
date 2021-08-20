import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import torch
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # This logger is required to build an engine

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

    def get_engine(self,path):
        with open(path,'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            return engine
    
    def build_engine(self):
        raise NotImplementedError

    def allocate_buffers(self,engine,context):
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

    def do_inference(self):
        raise NotImplementedError 

class EncWrapper(TrtWrapper):
    def __init__(self,trt_path):
        super(EncWrapper,self).__init__(trt_path)
    
    def do_inference(self, src, src_mask, batch_size=1):
        with self.get_engine(self.trt_path) as engine, engine.create_execution_context() as context:
            context.active_optimization_profile = 0
            shapes = [src.shape, src_mask.shape]
            # reshape bindings before doing inference
            for i,shape in enumerate(shapes):
                binding_shape = context.get_binding_shape(i)
                for j in range(len(shape)):
                    binding_shape[j] = shape[j]
                context.set_binding_shape(i,(binding_shape))
            inputs, outputs, bindings, stream = self.allocate_buffers(engine,context)
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
        with self.get_engine(self.trt_path) as engine, engine.create_execution_context() as context:
            context.active_optimization_profile = 0
            shapes = [decoder_states.shape, decoder_mask.shape, encoder_states.shape, encoder_mask.shape]
            # reshape bindings before doing inference
            for i,shape in enumerate(shapes):
                binding_shape = context.get_binding_shape(i)
                for j in range(len(shape)):
                    binding_shape[j] = shape[j]
                context.set_binding_shape(i,(binding_shape))
            inputs, outputs, bindings, stream = self.allocate_buffers(engine,context)
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
        with self.get_engine(self.trt_path) as engine, engine.create_execution_context() as context:
            context.active_optimization_profile = 0
            shapes = [decoder_states.shape, decoder_mask.shape, encoder_states.shape, encoder_mask.shape, decoder_mems.shape]
            # reshape bindings before doing inference
            for i,shape in enumerate(shapes):
                binding_shape = context.get_binding_shape(i)
                for j in range(len(shape)):
                    binding_shape[j] = shape[j]
                context.set_binding_shape(i,(binding_shape))
            inputs, outputs, bindings, stream = self.allocate_buffers(engine,context)
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

