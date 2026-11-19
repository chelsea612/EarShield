"""Test audio for the conversation pipeline: back-to-back turns, overlapping speech, and room versions.

Usage: python make_fixtures_v2.py <test-other dir> <train-clean-100 dir> <out dir> <hf model dir>
"""
import glob, json, sys
import numpy as np, librosa, soundfile as sf, onnxruntime as ort, kaldi_native_fbank as knf

SR = 16000
lib, imp_dir, out, hf = sys.argv[1:5]
rng = np.random.default_rng(7)

def speech(spk, start_file, seconds):
    files = sorted(glob.glob(f"{lib}/{spk}/*.flac"))[start_file:]
    w = np.concatenate([librosa.effects.trim(librosa.load(f, sr=SR)[0], top_db=30)[0] for f in files])
    return w[: int(seconds * SR)].astype(np.float32)

FRIEND, BLOCKED = "1688", "2414"

# Living-room simulation: mild echo + two background talkers 15 dB down (different people).
babble_src = [librosa.load(f, sr=SR)[0] for f in sorted(glob.glob(f"{imp_dir}/*.flac"))[-4:]]
def rir(rt60=0.4, drr_db=3):
    n = int(rt60 * SR); t = np.arange(1, n) / SR
    tail = rng.standard_normal(n - 1) * np.exp(-6.9 * t / rt60)
    tail *= 10 ** (-drr_db / 20) / np.sqrt(np.sum(tail ** 2))
    return np.concatenate([[1.0], tail])
def room(w, snr_db=15):
    y = np.convolve(w, rir())[: len(w)]
    noise = np.zeros(len(y))
    for b in babble_src[:2]:
        src = np.tile(b, len(y) // len(b) + 2); off = rng.integers(0, len(b)); noise += src[off:off + len(y)]
    speech_rms = np.sqrt(np.mean(y[np.abs(y) > 1e-4] ** 2))
    noise *= speech_rms / np.sqrt(np.mean(noise ** 2)) * 10 ** (-snr_db / 20)
    return (y + noise).astype(np.float32)

def save(name, w, labels):
    w = (w / max(1e-9, np.abs(w).max()) * 0.5).astype(np.float32)
    sf.write(f"{out}/{name}.wav", w, SR, subtype="PCM_16")
    json.dump(labels, open(f"{out}/{name}_labels.json", "w"), indent=1)

sources = {}
def sequence(parts):
    """parts: list of (who, audio, start_seconds). Overlaps allowed. Returns mix and per-sample labels.
    Also keeps each person's track in `sources` so tests can measure what's left of each voice."""
    end = max(int(s * SR) + len(a) for _, a, s in parts)
    mix = np.zeros(end, np.float32)
    labels = []
    sources.clear()
    for name, a, s in parts:
        i = int(s * SR); mix[i:i + len(a)] += a
        sources.setdefault(name, np.zeros(end, np.float32))[i:i + len(a)] += a
        labels.append({"who": name, "start": i, "end": i + len(a)})
    return mix, labels

enroll = speech(BLOCKED, 0, 15)
sf.write(f"{out}/blocked_enroll.wav", enroll, SR, subtype="PCM_16")

# Back-to-back turns, no pauses.
turns, t = [], 0.0
for who, spk, f, secs in [("friend", FRIEND, 4, 4), ("blocked", BLOCKED, 4, 4), ("friend", FRIEND, 6, 3),
                          ("blocked", BLOCKED, 7, 3), ("friend", FRIEND, 8, 4)]:
    a = speech(spk, f, secs) * 0.5
    turns.append((who, a, t)); t += len(a) / SR
mix, labels = sequence(turns)
save("turns", mix, labels)
save("turns_room", room(mix), labels)

# Talking over each other: friend 0-8 s, blocked 4-10 s (overlap 4-8 s), friend alone again 11-15 s.
parts = [("friend", speech(FRIEND, 4, 8) * 0.5, 0.0), ("blocked", speech(BLOCKED, 4, 6) * 0.5, 4.0),
         ("friend", speech(FRIEND, 9, 4) * 0.5, 11.0)]
mix, labels = sequence(parts)
peak = np.abs(mix).max()
for who, src in sources.items():  # same scaling as save() applies to the mix
    sf.write(f"{out}/overlap_src_{who}.wav", (src / peak * 0.5).astype(np.float32), SR, subtype="PCM_16")
save("overlap", mix, labels)
save("overlap_room", room(mix), labels)
sf.write(f"{out}/blocked_enroll_room.wav", room(enroll), SR, subtype="PCM_16")

# References for the Kotlin feature and model tests.
clip = sf.read(f"{out}/blocked_enroll.wav", dtype="float32")[0][:3 * SR]
o = knf.FbankOptions(); o.frame_opts.dither = 0; o.frame_opts.window_type = "hamming"; o.mel_opts.num_bins = 80
f = knf.OnlineFbank(o); f.accept_waveform(SR, (clip * 32768).tolist()); f.input_finished()
fb = np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)
fb_cmn = fb - fb.mean(0, keepdims=True)
rn = ort.InferenceSession(f"{hf}/resnet34_LM.onnx")
emb = rn.run(None, {"feats": fb_cmn[None]})[0][0]; emb /= np.linalg.norm(emb)
seg = ort.InferenceSession(f"{hf}/pyannote_seg.onnx")
ov = sf.read(f"{out}/overlap.wav", dtype="float32")[0][:160000]
y = seg.run(None, {"x": ov[None, None]})[0][0]
json.dump({"fbank_raw": fb.tolist(), "resnet_embedding": emb.tolist(), "seg_argmax": y.argmax(-1).tolist()},
          open(f"{out}/reference_v2.json", "w"))
print("frames", fb.shape, "seg frames", y.shape, "classes used", sorted(set(y.argmax(-1).tolist())))

# --- Three people: turns, quick back-and-forth and overlaps (for learning voices from a conversation) ---
A, B, C = FRIEND, BLOCKED, "3005"
plan = [  # (who, speaker, file index, seconds, start)
    ("A", A, 1, 5, 0.0), ("B", B, 1, 4, 5.0), ("C", C, 1, 5, 9.0),
    ("A", A, 2, 4, 13.0),                       # A starts while C finishes (1 s overlap)
    ("B", B, 2, 5, 17.5), ("C", C, 2, 3, 19.0),  # C talks over B
    ("A", A, 3, 5, 23.0), ("B", B, 3, 4, 28.5), ("C", C, 3, 5, 33.0),
    ("A", A, 5, 3, 37.0),                        # A interrupts C
    ("B", B, 5, 5, 40.5), ("C", C, 5, 4, 46.0),
]
parts = [(who, speech(spk, f, secs) * 0.5, start) for who, spk, f, secs, start in plan]
mix, labels = sequence(parts)
save("three_people", mix, labels)
save("three_people_room", room(mix), labels)
sf.write(f"{out}/friend_enroll.wav", speech(FRIEND, 0, 15), SR, subtype="PCM_16")
print("three_people:", len(mix) / SR, "s")
