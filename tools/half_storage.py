"""Store an ONNX model's large weights as float16 to halve the file, without changing how it runs.

Each big float32 weight becomes a float16 weight plus a Cast back to float32. ONNX Runtime folds those
Casts when it loads the model, so inference still runs in float32 at full speed.

Usage: python half_storage.py in.onnx out.onnx
"""
import sys
import numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto

model = onnx.load(sys.argv[1])
g = model.graph
new_inits, casts = [], []
for init in list(g.initializer):
    if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= 1024:
        arr = numpy_helper.to_array(init)
        half = numpy_helper.from_array(arr.astype(np.float16), init.name + "__fp16")
        new_inits.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
        g.initializer.remove(init)
g.initializer.extend(new_inits)
for i, c in enumerate(casts):
    g.node.insert(i, c)
onnx.checker.check_model(model)
onnx.save(model, sys.argv[2])
print(f"{len(casts)} weights stored as float16")
