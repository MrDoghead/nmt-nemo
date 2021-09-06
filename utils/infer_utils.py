import torch
import time

class MeasureTime():
    def __init__(self, measurements, key, cpu_run=False):
        self.measurements = measurements
        self.key = key
        self.cpu_run = cpu_run

    def __enter__(self):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.cpu_run:
            torch.cuda.synchronize()
        dtime = time.perf_counter() - self.t0
        if self.key in self.measurements:
            self.measurements[self.key].append(dtime)
        else:
            self.measurements[self.key] = [dtime]
