"""Compare voice encoders in a simulated room: echo + background chatter, unseen impostor speakers.

Usage: python compare_encoders.py <librispeech test-other dir> <train-clean-100 dir> <ge2e.onnx> <resnet34.onnx> <campplus.onnx>
"""
import glob, os, sys
import numpy as np, librosa, onnxruntime as ort, kaldi_native_fbank as knf

SR = 16000
rng = np.random.default_rng(0)
test_dir, imp_dir, ge2e_p, rn_p, cam_p = sys.argv[1:6]

def load(f):
    w = librosa.load(f, sr=SR)[0]
    return librosa.effects.trim(w, top_db=30)[0]

# ---- simulated room -----------------------------------------------------------------------------
def rir(rt60, drr_db):
    """Exponentially decaying echo tail, with the direct sound [drr_db] louder than the echoes."""
    n = int(rt60 * SR)
    t = np.arange(1, n) / SR
    tail = rng.standard_normal(n - 1) * np.exp(-6.9 * t / rt60)
    tail *= 10 ** (-drr_db / 20) / np.sqrt(np.sum(tail ** 2))
    return np.concatenate([[1.0], tail])

COND = dict(rt60=0.4, drr=3, snr=15, talkers=2)

imp_files = sorted(glob.glob(f"{imp_dir}/*.flac"))
babble_src = [load(f) for f in imp_files[-12:]]  # chatter speakers, kept out of the impostor set
imp_files = imp_files[:-12]

def babble(n):
    out = np.zeros(n)
    for w in rng.choice(len(babble_src), COND["talkers"], replace=False):
        src = np.tile(babble_src[w], n // len(babble_src[w]) + 2)
        off = rng.integers(0, len(babble_src[w]))
        out += src[off:off + n]
    return out

def room(w):
    if COND is None: return (w / (np.abs(w).max() + 1e-9) * 0.3).astype(np.float32)
    snr_db = COND["snr"]
    y = np.convolve(w, rir(COND["rt60"], COND["drr"]))[: len(w)]
    noise = babble(len(y))
    noise *= np.sqrt(np.mean(y ** 2) / (np.mean(noise ** 2) + 1e-12)) * 10 ** (-snr_db / 20)
    y = y + noise
    return (y / (np.abs(y).max() + 1e-9) * 0.3).astype(np.float32)

# ---- encoders -----------------------------------------------------------------------------------
ge2e = ort.InferenceSession(ge2e_p); rn = ort.InferenceSession(rn_p); cam = ort.InferenceSession(cam_p)

def ge2e_embed(ws):
    def prep(w):
        db = 10 * np.log10(np.mean(w ** 2) + 1e-12); g = -30 - db
        w = w * 10 ** (g / 20) if g > 0 else w
        return librosa.feature.melspectrogram(y=w, sr=SR, n_fft=400, hop_length=160, n_mels=40).astype(np.float32).T
    return np.concatenate([ge2e.run(None, {"mels": prep(w)[None]})[0] for w in ws])

def fbank(w, window, scale):
    o = knf.FbankOptions()
    o.frame_opts.dither = 0
    o.frame_opts.window_type = window
    o.frame_opts.samp_freq = SR
    o.mel_opts.num_bins = 80
    f = knf.OnlineFbank(o)
    f.accept_waveform(SR, (w * scale).tolist()); f.input_finished()
    x = np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)
    return x - x.mean(0, keepdims=True)

def unit(e): return e / np.linalg.norm(e, axis=-1, keepdims=True)

def rn_embed(ws): return unit(np.concatenate([rn.run(None, {"feats": fbank(w, "hamming", 32768)[None]})[0] for w in ws]))
def cam_embed(ws): return unit(np.concatenate([cam.run(None, {"x": fbank(w, "povey", 1.0)[None]})[0] for w in ws]))

encoders = {"GE2E (current)": ge2e_embed, "ResNet34": rn_embed, "CAM++": cam_embed}

CONDITIONS = {
    "clean": None,
    "living room (echo, 2 people chatting 15 dB down)": dict(rt60=0.4, drr=3, snr=15, talkers=2),
    "busy cafe (more echo, 4 people chatting 8 dB down)": dict(rt60=0.6, drr=0, snr=8, talkers=4),
}
raw = {}
for cname, cond in CONDITIONS.items():
    COND = cond
    print(f"\n== {cname}", flush=True)
    # ---- data ---------------------------------------------------------------------------------------
    speakers = sorted(d for d in os.listdir(test_dir) if d.isdigit())
    enroll, targets = {}, {}
    for s in speakers:
        if s not in raw: raw[s] = np.concatenate([load(f) for f in sorted(glob.glob(f"{test_dir}/{s}/*.flac"))])
        w = raw[s]
        w = room(w)
        enroll[s] = w[: 12 * SR]; targets[s] = w[12 * SR:]
    if "imp" not in raw: raw["imp"] = [load(f) for f in imp_files]
    impostors = [room(w) for w in raw["imp"]]

    def chop(w, n, k):
        return [w[i:i + n] for i in range(0, len(w) - n, n)][:k]

    for name, emb in encoders.items():
        cents = {s: unit(emb(chop(enroll[s], int(1.6 * SR), 20)).mean(0)) for s in speakers}
        C = np.stack([cents[s] for s in speakers])
        line = f"{name:15s}"
        for secs in (1.0, 2.0, 3.0):
            n = int(secs * SR)
            tgt = []
            for i, s in enumerate(speakers):
                E = emb(chop(targets[s], n, 15))
                tgt += list(E @ cents[s])
            imp = []
            for w in impostors:
                parts = chop(w, n, 2)
                if not parts: continue
                E = emb(parts)
                imp += list((E @ C.T).max(1))  # best match against any enrolled voice = a false-block chance
            tgt, imp = np.array(tgt), np.array(imp)
            thr = np.quantile(tgt, 0.05)  # threshold that catches 95% of the blocked person
            fpr = (imp >= thr).mean()
            ths = np.linspace(-1, 1, 2001)
            eer = min(ths, key=lambda t: abs((tgt < t).mean() - (imp >= t).mean()))
            eerv = ((tgt < eer).mean() + (imp >= eer).mean()) / 2
            line += f" | {secs:.0f}s: EER {eerv*100:4.1f}%  wrong-blocks@95%catch {fpr*100:4.1f}%"
        print(line, flush=True)
