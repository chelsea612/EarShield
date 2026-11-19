"""How well does GTCRN pull speech out of traffic, hiss and distant chatter, and does "mix minus speech"
give a clean background bed (for covering a blocked voice with the room's own sound)?"""
import glob, sys, time
import numpy as np, librosa, onnxruntime as ort
SR = 16000
lib, imp_dir, model = sys.argv[1:4]
rng = np.random.default_rng(5)
so = ort.SessionOptions(); so.intra_op_num_threads = 1
sess = ort.InferenceSession(model, so)
win = np.sqrt(np.hanning(513)[:512]).astype(np.float32)  # periodic sqrt-Hann, like torch.hann_window(512).pow(0.5)

def enhance(x):
    """Streaming GTCRN: one 512-sample frame every 256 samples, overlap-add."""
    conv = np.zeros((2, 1, 16, 16, 33), np.float32); tra = np.zeros((2, 3, 1, 1, 16), np.float32)
    inter = np.zeros((2, 1, 33, 16), np.float32)
    pad = np.concatenate([np.zeros(256, np.float32), x.astype(np.float32), np.zeros(512, np.float32)])
    out = np.zeros(len(pad) + 512, np.float32)
    for start in range(0, len(pad) - 512 + 1, 256):
        spec = np.fft.rfft(pad[start:start + 512] * win)
        mix = np.stack([spec.real, spec.imag], -1).astype(np.float32)[None, :, None, :]
        enh, conv, tra, inter = sess.run(None, {"mix": mix, "conv_cache": conv, "tra_cache": tra, "inter_cache": inter})
        e = enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]
        out[start:start + 512] += np.fft.irfft(e, 512).astype(np.float32) * win
    return out[256:256 + len(x)]

def sisdr(est, ref):
    a = est @ ref / (ref @ ref) * ref; return 10 * np.log10((a @ a) / ((est - a) @ (est - a)))
def load(f): return librosa.effects.trim(librosa.load(f, sr=SR)[0], top_db=30)[0]

speech = np.concatenate([load(f) for f in sorted(glob.glob(f"{lib}/1688/*.flac"))[:4]])[:12 * SR]
n = len(speech)
t = np.arange(n) / SR
def brown(n):
    w = np.cumsum(rng.standard_normal(n)); w -= np.convolve(w, np.ones(4000) / 4000, "same"); return w
def traffic(n):  # low rumble plus cars passing (slow swells of band-limited noise)
    rumble = brown(n)
    swell = np.clip(np.sin(2 * np.pi * t / 6.0), 0, None) ** 2 + 0.3
    hiss = np.convolve(rng.standard_normal(n), np.hanning(9) / 4.5, "same")
    return rumble / rumble.std() + 0.6 * swell * hiss / hiss.std()
def babble(n):  # three far-away talkers with echo
    out = np.zeros(n)
    for f in sorted(glob.glob(f"{imp_dir}/*.flac"))[:3]:
        w = librosa.load(f, sr=SR)[0]; w = np.tile(w, n // len(w) + 1)[:n]
        h = rng.standard_normal(int(0.5 * SR)) * np.exp(-6.9 * np.arange(int(0.5 * SR)) / (0.5 * SR)); h[0] += 2
        out += np.convolve(w, h)[:n]
    return out
noises = {"traffic": traffic(n), "hiss (white)": rng.standard_normal(n), "distant chatter": babble(n)}
for name, noise in noises.items():
    for snr in (10, 0):
        nz = noise / noise.std() * speech.std() * 10 ** (-snr / 20)
        mix = (speech + nz).astype(np.float32)
        tt = time.perf_counter(); enh = enhance(mix); el = time.perf_counter() - tt
        bg = mix - enh
        # Speech left in the background bed, relative to the original speech level:
        a = bg @ speech / (speech @ speech)
        print(f"{name:16s} at {snr:2d} dB SNR: speech clarity {sisdr(mix, speech):5.1f} -> {sisdr(enh, speech):5.1f} dB | "
              f"noise level after: {20*np.log10(np.std(enh - (enh @ speech / (speech @ speech)) * speech) / np.std(nz)):5.1f} dB | "
              f"speech left in background bed: {20*np.log10(abs(a) + 1e-6):5.1f} dB | {el / (n / SR) * 1000:.0f} ms per s")

print("\nSeparately: how much of each sound gets through on its own (0 dB = all of it)")
def level(x): return 20 * np.log10(np.std(x) + 1e-12)
sp = speech.astype(np.float32) * 0.1
print(f"  speech alone: {level(enhance(sp)) - level(sp):6.1f} dB")
for name, noise in list(noises.items()) + [("traffic, speech band only", None)]:
    if noise is None:  # traffic noise high-passed at 60 Hz (what you'd actually hear from a road)
        noise = noises["traffic"]; spec = np.fft.rfft(noise); f = np.fft.rfftfreq(len(noise), 1 / SR); spec[f < 60] = 0; noise = np.fft.irfft(spec, len(noise))
    nz = (noise / noise.std() * 0.03).astype(np.float32)
    print(f"  {name:26s}: {level(enhance(nz)) - level(nz):6.1f} dB")
