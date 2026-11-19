"""Pick thresholds for the ResNet34 pipeline from LibriSpeech in a simulated living room and cafe."""
import glob, os, sys
import numpy as np, librosa, onnxruntime as ort, kaldi_native_fbank as knf
SR = 16000
test_dir, imp_dir, rn_p = sys.argv[1:4]
rng = np.random.default_rng(1)
rn = ort.InferenceSession(rn_p)
def load(f): return librosa.effects.trim(librosa.load(f, sr=SR)[0], top_db=30)[0]
imp_files = sorted(glob.glob(f"{imp_dir}/*.flac"))
babble_src = [load(f) for f in imp_files[-12:]]; imp_files = imp_files[:-12]
def rir(rt60, drr):
    n = int(rt60 * SR); t = np.arange(1, n) / SR
    tail = rng.standard_normal(n - 1) * np.exp(-6.9 * t / rt60); tail *= 10 ** (-drr / 20) / np.sqrt(np.sum(tail ** 2))
    return np.concatenate([[1.0], tail])
def room(w, c):
    y = np.convolve(w, rir(c["rt60"], c["drr"]))[: len(w)]; nz = np.zeros(len(y))
    for i in rng.choice(len(babble_src), c["talkers"], replace=False):
        b = babble_src[i]; src = np.tile(b, len(y) // len(b) + 2); o = rng.integers(0, len(b)); nz += src[o:o + len(y)]
    nz *= np.sqrt(np.mean(y ** 2) / np.mean(nz ** 2)) * 10 ** (-c["snr"] / 20)
    return (y + nz).astype(np.float32)
def emb(w):
    o = knf.FbankOptions(); o.frame_opts.dither = 0; o.frame_opts.window_type = "hamming"; o.mel_opts.num_bins = 80
    f = knf.OnlineFbank(o); f.accept_waveform(SR, (w * 32768).tolist()); f.input_finished()
    x = np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32); x -= x.mean(0)
    e = rn.run(None, {"feats": x[None]})[0][0]; return e / np.linalg.norm(e)
def chop(w, n, k): return [w[i:i + n] for i in range(0, len(w) - n, n)][:k]
def consistency(w):
    parts = [emb(p) for p in chop(w, 3 * SR, 99)]  # 3 s slices (no overlap for speed)
    c = np.mean(parts, 0); c /= np.linalg.norm(c); return float(np.mean([p @ c for p in parts]))
speakers = sorted(d for d in os.listdir(test_dir) if d.isdigit())
raw = {s: np.concatenate([load(f) for f in sorted(glob.glob(f"{test_dir}/{s}/*.flac"))]) for s in speakers}
imps = [load(f) for f in imp_files[:120]]
for name, c in {"living room": dict(rt60=0.4, drr=3, snr=15, talkers=2),
                "cafe": dict(rt60=0.6, drr=0, snr=8, talkers=4)}.items():
    W = {s: room(raw[s], c) for s in speakers}
    cents = {}
    for s in speakers:
        e = np.mean([emb(p) for p in chop(W[s][:15 * SR], 3 * SR, 9)], 0); cents[s] = e / np.linalg.norm(e)
    C = np.stack([cents[s] for s in speakers])
    tgt, pair_same = [], []
    for s in speakers:
        E = [emb(p) for p in chop(W[s][15 * SR:], 3 * SR, 10)]
        tgt += [e @ cents[s] for e in E]
        pair_same += [E[i] @ E[i + 1] for i in range(len(E) - 1)]
    imp, pair_diff = [], []
    IE = []
    for w in imps:
        p = chop(room(w, c), 3 * SR, 1)
        if p: IE.append(emb(p[0]))
    IE = np.array(IE)
    imp = (IE @ C.T).max(1)
    pair_diff = [IE[i] @ IE[i + 1] for i in range(len(IE) - 1)]
    tgt, imp = np.array(tgt), np.array(imp)
    print(f"== {name}")
    print("  blocked person vs their voiceprint: 5%%=%.2f 25%%=%.2f median=%.2f" % tuple(np.quantile(tgt, [.05, .25, .5])))
    print("  others' best match to any voiceprint: median=%.2f 95%%=%.2f 99%%=%.2f max=%.2f" % tuple(np.quantile(imp, [.5, .95, .99, 1])))
    for th in (0.40, 0.45, 0.50, 0.55, 0.60):
        print(f"    threshold {th:.2f}: catches {(tgt >= th).mean()*100:5.1f}%  wrong-blocks {(imp >= th).mean()*100:4.1f}%")
    print("  3s vs 3s same person: 5%%=%.2f median=%.2f | different people: median=%.2f 95%%=%.2f 99%%=%.2f" % (
        np.quantile(pair_same, .05), np.median(pair_same), np.median(pair_diff), np.quantile(pair_diff, .95), np.quantile(pair_diff, .99)))
    one = [consistency(W[s][:15 * SR]) for s in speakers]
    mixed = [consistency(np.concatenate([W[a][:7 * SR], W[b][:8 * SR]])) for a, b in zip(speakers, speakers[1:] + speakers[:1])]
    print("  recording consistency: one voice min=%.2f median=%.2f | two voices median=%.2f max=%.2f" % (
        min(one), np.median(one), np.median(mixed), max(mixed)), flush=True)
