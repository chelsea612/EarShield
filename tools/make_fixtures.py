"""Create reference data and a test conversation for the Kotlin unit tests."""
import glob, json, sys
import numpy as np, librosa, soundfile as sf, onnxruntime as ort

SR = 16000
lib = sys.argv[1]; out = sys.argv[2]; models = sys.argv[3]

def speech(spk, start_file, seconds):
    files = sorted(glob.glob(f"{lib}/{spk}/*.flac"))[start_file:]
    w = np.concatenate([librosa.effects.trim(librosa.load(f, sr=SR)[0], top_db=30)[0] for f in files])
    return w[: int(seconds * SR)]

# Speaker 1688 = "friend" (allowed), 2414 = "blocked person"
friend_enroll = speech("1688", 0, 15)      # not used for enrollment, just spare
blocked_enroll = speech("2414", 0, 15)
sil = np.zeros(int(0.8 * SR), np.float32)
parts = [("friend", speech("1688", 4, 6)), ("gap", sil), ("blocked", speech("2414", 4, 6)), ("gap", sil),
         ("friend", speech("1688", 7, 6)), ("gap", sil), ("blocked", speech("2414", 7, 4)), ("gap", sil),
         ("friend", speech("1688", 9, 4))]
conv, labels, pos = [], [], 0
for name, w in parts:
    w = (w * 0.5).astype(np.float32)  # quieter than full scale, like a room mic
    labels.append({"who": name, "start": pos, "end": pos + len(w)})
    conv.append(w); pos += len(w)
conv = np.concatenate(conv)
sf.write(f"{out}/blocked_enroll.wav", blocked_enroll, SR, subtype="PCM_16")
sf.write(f"{out}/conversation.wav", conv, SR, subtype="PCM_16")
json.dump(labels, open(f"{out}/conversation_labels.json", "w"), indent=1)

# Reference mel filterbank and spectrogram for a 1 s clip (read back from the 16-bit file)
clip = sf.read(f"{out}/blocked_enroll.wav", dtype="float32")[0][:SR]
fb = librosa.filters.mel(sr=SR, n_fft=400, n_mels=40)
mel = librosa.feature.melspectrogram(y=clip, sr=SR, n_fft=400, hop_length=160, n_mels=40).T
enc = ort.InferenceSession(f"{models}/voice_encoder.onnx")
emb = enc.run(None, {"mels": mel[None].astype(np.float32)})[0][0]
# Silero VAD probabilities for the first 2 s of the conversation, using the v5 context scheme
vad = ort.InferenceSession(f"{models}/silero_vad.onnx")
conv16 = sf.read(f"{out}/conversation.wav", dtype="float32")[0]
state, ctx, probs = np.zeros((2, 1, 128), np.float32), np.zeros((1, 64), np.float32), []
for i in range(0, 2 * SR, 512):
    x = np.concatenate([ctx, conv16[None, i:i + 512]], axis=1)
    p, state = vad.run(None, {"input": x, "state": state, "sr": np.array(SR, np.int64)})
    probs.append(float(p[0, 0])); ctx = x[:, -64:]
json.dump({"filterbank": fb.tolist(), "mel": mel.tolist(), "embedding_no_norm": emb.tolist(),
           "vad_probs": probs}, open(f"{out}/reference.json", "w"))
print("conversation", len(conv) / SR, "s;", [(l["who"], round((l["end"]-l["start"])/SR,1)) for l in labels])
