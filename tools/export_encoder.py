"""Convert Resemblyzer's GE2E voice encoder to ONNX and check it against a numpy reference.

Usage: python export_encoder.py <pretrained.pt> <out.onnx>
"""
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
from load_legacy_pt import load

H, LAYERS, NMELS = 256, 3, 40

def reorder_gates(w):  # PyTorch i,f,g,o  ->  ONNX i,o,f,c
    i, f, g, o = np.split(w, 4, axis=0)
    return np.concatenate([i, o, f, g], axis=0)

def build(sd):
    inits, nodes = [], []
    nodes.append(helper.make_node("Transpose", ["mels"], ["x0"], perm=[1, 0, 2]))  # [T,B,40]
    x = "x0"
    for l in range(LAYERS):
        W = reorder_gates(sd[f"lstm.weight_ih_l{l}"])[None]
        R = reorder_gates(sd[f"lstm.weight_hh_l{l}"])[None]
        B = np.concatenate([reorder_gates(sd[f"lstm.bias_ih_l{l}"]),
                            reorder_gates(sd[f"lstm.bias_hh_l{l}"])])[None]
        for n, a in (("W", W), ("R", R), ("B", B)):
            inits.append(numpy_helper.from_array(a.astype(np.float32), f"{n}{l}"))
        nodes.append(helper.make_node("LSTM", [x, f"W{l}", f"R{l}", f"B{l}"], [f"y{l}", f"yh{l}"],
                                      hidden_size=H))
        if l < LAYERS - 1:
            inits.append(numpy_helper.from_array(np.array([1], np.int64), f"ax{l}"))
            nodes.append(helper.make_node("Squeeze", [f"y{l}", f"ax{l}"], [f"x{l+1}"]))  # [T,B,H]
            x = f"x{l+1}"
    inits.append(numpy_helper.from_array(np.array([0], np.int64), "ax_last"))
    nodes.append(helper.make_node("Squeeze", [f"yh{LAYERS-1}", "ax_last"], ["hidden"]))  # [B,H]
    inits.append(numpy_helper.from_array(sd["linear.weight"].astype(np.float32), "lin_w"))
    inits.append(numpy_helper.from_array(sd["linear.bias"].astype(np.float32), "lin_b"))
    nodes.append(helper.make_node("Gemm", ["hidden", "lin_w", "lin_b"], ["raw"], transB=1))
    nodes.append(helper.make_node("Relu", ["raw"], ["pos"]))
    nodes.append(helper.make_node("ReduceL2", ["pos"], ["norm"], axes=[1], keepdims=1))
    inits.append(numpy_helper.from_array(np.array(1e-8, np.float32), "eps"))
    nodes.append(helper.make_node("Max", ["norm", "eps"], ["norm_safe"]))
    nodes.append(helper.make_node("Div", ["pos", "norm_safe"], ["embedding"]))
    graph = helper.make_graph(
        nodes, "ge2e_voice_encoder",
        [helper.make_tensor_value_info("mels", TensorProto.FLOAT, ["batch", "frames", NMELS])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", H])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                              producer_name="earshield")
    model.ir_version = 8  # loadable by ONNX Runtime 1.14+
    onnx.checker.check_model(model)
    return model

def numpy_reference(sd, mels):
    sig = lambda z: 1 / (1 + np.exp(-z))
    x = mels.astype(np.float64)  # [B,T,40]
    for l in range(LAYERS):
        Wi, Wh = sd[f"lstm.weight_ih_l{l}"], sd[f"lstm.weight_hh_l{l}"]
        b = sd[f"lstm.bias_ih_l{l}"] + sd[f"lstm.bias_hh_l{l}"]
        Bn, T, _ = x.shape
        h, c = np.zeros((Bn, H)), np.zeros((Bn, H))
        out = []
        for t in range(T):
            g = x[:, t] @ Wi.T + h @ Wh.T + b
            i, f, gg, o = np.split(g, 4, axis=1)
            c = sig(f) * c + sig(i) * np.tanh(gg)
            h = sig(o) * np.tanh(c)
            out.append(h)
        x = np.stack(out, axis=1)
    raw = np.maximum(h @ sd["linear.weight"].T + sd["linear.bias"], 0)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)

if __name__ == "__main__":
    sd = load(sys.argv[1])["model_state"]
    model = build(sd)
    onnx.save(model, sys.argv[2])
    sess = ort.InferenceSession(sys.argv[2])
    rng = np.random.default_rng(0)
    mels = (rng.random((2, 160, NMELS)) * 2).astype(np.float32)
    got = sess.run(None, {"mels": mels})[0]
    ref = numpy_reference(sd, mels)
    print("max abs diff vs numpy reference:", float(np.abs(got - ref).max()))
    assert np.abs(got - ref).max() < 1e-4
    print("OK ->", sys.argv[2])
