"""Check how well the encoder separates LibriSpeech speakers at various window lengths."""
import glob, os, sys
import numpy as np, librosa, onnxruntime as ort

SR = 16000
sess = ort.InferenceSession(sys.argv[1])
root = sys.argv[2]

def normalize(w, target=-30.0):
    rms_db = 10 * np.log10(np.mean(w ** 2) + 1e-12)
    gain = target - rms_db
    return w * 10 ** (gain / 20) if gain > 0 else w

def mel(w):
    return librosa.feature.melspectrogram(y=w, sr=SR, n_fft=400, hop_length=160, n_mels=40).astype(np.float32).T

def embed(windows):
    m = np.stack([mel(normalize(w)) for w in windows])
    return sess.run(None, {"mels": m})[0]

def unit(v): return v / np.linalg.norm(v)

speakers = sorted(d for d in os.listdir(root) if d.isdigit())
data = {}
for s in speakers:
    files = sorted(glob.glob(f"{root}/{s}/*.flac"))
    wavs = [librosa.load(f, sr=SR)[0] for f in files]
    wavs = [librosa.effects.trim(w, top_db=30)[0] for w in wavs]
    data[s] = wavs

# Enroll from first ~15 s of speech, sliced into 1.6 s partials (like a one-tap enrollment)
enrolled = {}
for s, wavs in data.items():
    enroll = np.concatenate(wavs)[: 15 * SR]
    parts = [enroll[i:i + 25600] for i in range(0, len(enroll) - 25600, 25600 // 2)]
    enrolled[s] = unit(embed(parts).mean(0))
    data[s] = np.concatenate(wavs)[15 * SR:]  # test on the rest

for secs in (0.6, 0.8, 1.0, 1.2, 1.6):
    n = int(secs * SR)
    same, diff, correct, total = [], [], 0, 0
    for s, w in data.items():
        wins = [w[i:i + n] for i in range(0, len(w) - n, n)][:20]
        if not wins: continue
        E = embed(wins)
        for e in E:
            sims = {t: float(e @ c) for t, c in enrolled.items()}
            same.append(sims[s]); diff += [v for t, v in sims.items() if t != s]
            correct += max(sims, key=sims.get) == s; total += 1
    same, diff = np.array(same), np.array(diff)
    best = max(np.arange(0.5, 0.95, 0.01), key=lambda th: ((same >= th).mean() + (diff < th).mean()))
    print(f"{secs:.1f}s windows: same={same.mean():.3f}±{same.std():.3f} diff={diff.mean():.3f}±{diff.std():.3f} "
          f"top1={correct/total:.1%} | thr {best:.2f}: catches {(same>=best).mean():.1%}, false-blocks {(diff>=best).mean():.1%}")
