"""Can Conv-TasNet pull Jerry out of a mix? Measures how much of the friend we keep and Jerry we remove.

For each test mix (friend + blocked person talking at once, 4 s), separate into two streams, pick the
stream that sounds least like the blocked person's voiceprint, and compare it to the friend alone.
"""
import glob, sys, time
import numpy as np, librosa, onnxruntime as ort, kaldi_native_fbank as knf
SR = 16000
lib, imp_dir, sep_p, rn_p = sys.argv[1:5]
rng = np.random.default_rng(3)
so = ort.SessionOptions(); so.intra_op_num_threads = 1
sep = ort.InferenceSession(sep_p, so); rn = ort.InferenceSession(rn_p)
print("separator inputs", [(i.name, i.shape) for i in sep.get_inputs()], "outputs", [(o.name, o.shape) for o in sep.get_outputs()])
def load(f): return librosa.effects.trim(librosa.load(f, sr=SR)[0], top_db=30)[0]
def emb(w):
    o = knf.FbankOptions(); o.frame_opts.dither = 0; o.frame_opts.window_type = "hamming"; o.mel_opts.num_bins = 80
    f = knf.OnlineFbank(o); f.accept_waveform(SR, (w * 32768).tolist()); f.input_finished()
    x = np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32); x -= x.mean(0)
    e = rn.run(None, {"feats": x[None]})[0][0]; return e / np.linalg.norm(e)
def sisdr(est, ref):
    a = est @ ref / (ref @ ref + 1e-9) * ref; return 10 * np.log10((a @ a) / ((est - a) @ (est - a) + 1e-9) + 1e-12)
def leak(est, friend, blocked):
    """Blocked person's level relative to the friend in [est], in dB (input mixes start at 0 dB)."""
    A = np.stack([friend, blocked], 1); (a, b), *_ = np.linalg.lstsq(A, est, rcond=None)
    return 20 * np.log10((abs(b) * np.linalg.norm(blocked)) / (abs(a) * np.linalg.norm(friend) + 1e-9) + 1e-9)
babble_src = [load(f) for f in sorted(glob.glob(f"{imp_dir}/*.flac"))[-4:]]
def rir(rt60=0.4, drr=3):
    n = int(rt60 * SR); t = np.arange(1, n) / SR
    tail = rng.standard_normal(n - 1) * np.exp(-6.9 * t / rt60); tail *= 10 ** (-drr / 20) / np.sqrt(np.sum(tail ** 2))
    return np.concatenate([[1.0], tail])
def babble(n):
    out = np.zeros(n)
    for b in babble_src[:2]:
        src = np.tile(b, n // len(b) + 2); o = rng.integers(0, len(b)); out += src[o:o + n]
    return out
spk = sorted(d.split("/")[-1] for d in glob.glob(f"{lib}/*") if d.split("/")[-1].isdigit())
voice = {s: np.concatenate([load(f) for f in sorted(glob.glob(f"{lib}/{s}/*.flac"))]) for s in spk}
N = 4 * SR
for cond in ("clean", "room"):
    res = []; t_sep = 0; secs = 0
    for i in range(len(spk)):
        fr, bl = spk[i], spk[(i + 3) % len(spk)]
        enroll_b = voice[bl][:15 * SR]
        f_src = voice[fr][20 * SR:20 * SR + N]; b_src = voice[bl][20 * SR:20 * SR + N]
        b_src = b_src * np.sqrt(np.mean(f_src ** 2) / np.mean(b_src ** 2))  # equal loudness
        if cond == "room":
            h1, h2 = rir(), rir()
            f_src = np.convolve(f_src, h1)[:N]; b_src = np.convolve(b_src, h2)[:N]; enroll_b = np.convolve(enroll_b, rir())[:len(enroll_b)]
            nz = babble(N); nz *= np.sqrt(np.mean((f_src + b_src) ** 2) / np.mean(nz ** 2)) * 10 ** (-15 / 20)
            mix = f_src + b_src + nz
        else:
            mix = f_src + b_src
        parts = [emb(enroll_b[j:j + 3 * SR]) for j in range(0, len(enroll_b) - 3 * SR, int(1.5 * SR))]
        vb = np.mean(parts, 0); vb /= np.linalg.norm(vb)
        scale = 0.5 / np.abs(mix).max(); m = (mix * scale).astype(np.float32)
        t = time.perf_counter(); est = sep.run(None, {sep.get_inputs()[0].name: m[None]})[0][0]; t_sep += time.perf_counter() - t; secs += N / SR
        scores = [float(emb(e.astype(np.float32)) @ vb) for e in est]
        keep = est[int(np.argmin(scores))] / scale
        correct = sisdr(keep, f_src) > sisdr(est[int(np.argmax(scores))] / scale, f_src)
        res.append((sisdr(mix, f_src), sisdr(keep, f_src), leak(mix, f_src, b_src), leak(keep, f_src, b_src), correct, min(scores), max(scores)))
    r = np.array(res, dtype=float)
    print(f"== {cond}: picked the right stream {r[:, 4].mean() * 100:.0f}% of the time | "
          f"friend clarity (SI-SDR) {r[:, 0].mean():.1f} dB -> {r[:, 1].mean():.1f} dB | "
          f"blocked person vs friend {r[:, 2].mean():.1f} dB -> {np.median(r[:, 3]):.1f} dB (median), worst {r[:, 3].max():.1f} dB | "
          f"stream scores vs voiceprint: kept {r[:, 5].mean():.2f}, removed {r[:, 6].mean():.2f} | "
          f"separation speed {t_sep / secs * 1000:.0f} ms per second of audio (1 thread)")
