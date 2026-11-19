"""Read a legacy (pre-1.6, non-zip) PyTorch checkpoint into numpy arrays without torch."""
import pickle, struct, sys
import numpy as np

DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
          "LongStorage": np.int64, "IntStorage": np.int32, "ByteStorage": np.uint8}

class _StorageType:
    def __init__(self, name): self.name = name

class _Unpickler(pickle.Unpickler):
    def __init__(self, f, storages):
        super().__init__(f); self.storages = storages
    def find_class(self, module, name):
        if module.startswith("torch") and name.endswith("Storage"):
            return _StorageType(name)
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            def rebuild(storage, offset, size, stride, *rest):
                return ("tensor", storage, offset, tuple(size), tuple(stride))
            return rebuild
        if module == "collections" and name == "OrderedDict":
            import collections; return collections.OrderedDict
        return super().find_class(module, name)
    def persistent_load(self, pid):
        _, stype, key, location, size, *_ = pid
        self.storages.setdefault(key, [stype.name, size, None])
        return key

def load(path):
    with open(path, "rb") as f:
        for _ in range(3):  # magic number, protocol version, sys info
            pickle.load(f)
        storages = {}
        obj = _Unpickler(f, storages).load()
        keys = pickle.load(f)
        for key in keys:
            n = struct.unpack("<q", f.read(8))[0]
            dt = DTYPES[storages[key][0]]
            storages[key][2] = np.frombuffer(f.read(n * np.dtype(dt).itemsize), dtype=dt)
    def materialize(x):
        if isinstance(x, tuple) and x and x[0] == "tensor":
            _, key, off, size, stride = x
            data = storages[key][2]
            itemsize = data.itemsize
            return np.lib.stride_tricks.as_strided(data[off:], shape=size,
                                                   strides=[s * itemsize for s in stride]).copy()
        if isinstance(x, dict):
            return {k: materialize(v) for k, v in x.items()}
        return x
    return materialize(obj)

if __name__ == "__main__":
    ckpt = load(sys.argv[1])
    sd = ckpt.get("model_state", ckpt)
    for k, v in sd.items():
        print(k, getattr(v, "shape", v))
