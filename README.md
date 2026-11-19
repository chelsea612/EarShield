# Earshield

An Android app that passes the room's sound through to your earbuds and removes the voices of people you've blocked. It recognizes *who* is talking, not *what* they say. When a blocked person talks over someone else, it separates the voices so you still hear the other person.

Everything runs on the phone. Only voiceprints are saved (256 numbers per person), never audio, and nothing is sent anywhere. Sound is only ever played to earbuds or headphones, never the phone's speaker.

## Install

1. Download [`Earshield-0.4.1.apk`](Earshield-0.4.1.apk) on your Android phone and open it. Android will ask you to allow installs from that app (Files, Chrome, etc.).
2. Open Earshield, tap **Start listening**, and allow microphone and notification access.

The APK is signed with a development key, so Play Protect may warn about an unknown developer. That's expected for a sideloaded test build.

## Use

1. **Start listening.** You hear the room through your earbuds, 3 s late by default.
2. **Add a voice → Record now** while only that person is talking: at least 6 s of speech, around 12 s is best. **Use last 10 s** learns whoever just spoke.
3. **Teach it my voice** (under *Never blocked*) so your own voice is never covered.
4. When a blocked voice is detected, it's covered (soft noise by default), and the phone buzzes once.
5. If someone is covered by mistake, tap **Not them** under *Recently covered*. That voice is then treated as allowed.

| Setting | What it does |
|---|---|
| Microphone: **Phone mic** | Full sound quality. Keep the phone in a shirt pocket or on a lanyard. |
| Microphone: **Earbud mic** | Uses the earbuds' mic. Bluetooth switches to call mode: mono, call-quality sound in both directions. |
| **Delay** | How late you hear the room (1–5 s). More time means better decisions. |
| **Match needed** | How similar a voice must be to block it (default 42%). Raise it if friends get covered; lower it if a blocked person gets through. |
| **When they talk over someone** | **Separate** (default) pulls their voice out of the mix; **Turn down** lowers both; **Cover** covers both. |
| **Cover with** | Soft noise, silence, or quieter. |
| **Pause blocking** | Lets everything through, for when you need to hear a blocked person. |

## How it works

Every second, on a background thread, for the last 10 s of audio:

1. **Who talks when:** [pyannote segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) marks when each of up to 3 people talks, including overlaps.
2. **Who they are:** [WeSpeaker ResNet34](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) turns up to 3 s of each person's solo speech into a voiceprint. Voiceprints are followed from one check to the next, so each person builds up evidence over the conversation and a voice change under the same label is caught.
3. **Decide:** each person is compared with the blocked and allowed voices. Audio that hasn't played yet is covered where a blocked person talks.
4. **Separate overlaps:** where a blocked person talks over someone, [Conv-TasNet](https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k) splits the audio into two voices, and the one that isn't them is played instead. If the split isn't clear, the overlap is turned down.

Each stretch of audio is re-checked several times before it plays, and the latest, best-informed decision wins. [Silero VAD](https://github.com/snakers4/silero-vad) handles enrollment and the mic meter.

## Test results

`./gradlew :core:test` runs the real pipeline on LibriSpeech conversations, including simulated rooms with echo and background chatter. With one speaker blocked:

| Test | Blocked person heard | Other person kept |
|---|---|---|
| Turns with pauses | 0% | 100% |
| Back-to-back turns, no pauses | 3% | 96% |
| Talking over each other, separated | 28 dB below the other person | 76% |
| Talking over each other, turned down | 17 dB below the other person | 58% |

In a simulated living room, the voice model wrongly blocked 0% of other speakers on 3 s clips while catching 95% of the blocked one. The previous model wrongly blocked 38–74%. Loud background chatter can still hide a blocked voice completely: one test turn scored far below any usable threshold even offline. The tests also confirm the Kotlin features and models match their Python originals.

**Not yet tested on a real phone.**

## Known limits

- **Noisy places:** chatter loud enough to bury a voice can stop it being recognized.
- **Separation:** the model was trained on clean speech, so the other voice can sound warbly, especially with echo. It picked the right voice 100% of the time on clean audio and 90% in a simulated room.
- **Phone speed:** separation is costly. If the phone can't keep up, the app warns you, and some audio plays unchecked. A longer delay or **Turn down** helps.
- **Sound around the earbud tips:** earbuds without noise cancelling let some room sound in directly.

## Project layout

```
core/    Platform-independent Kotlin: Kaldi fbank features, ONNX models, the blocking pipeline, tests
app/     Android app: audio engine (mic routing, private-output guard), foreground service, Compose UI
models/  The ONNX models bundled into the APK, with their licenses
tools/   Python scripts for model comparison, calibration, and test data
```

## Building

Requires JDK 17+ and the Android SDK (platform 36). Open the folder in Android Studio, or run:

```bash
./gradlew :core:test :app:assembleRelease
```

## Credits and licenses

- Silero VAD: MIT. pyannote segmentation-3.0: MIT.
- WeSpeaker ResNet34-LM: CC BY 4.0.
- Conv-TasNet (Libri2Mix) by Joris Cosentino / Asteroid: CC BY-SA 4.0.
- Test audio is from [LibriSpeech](https://www.openslr.org/12) (CC BY 4.0), taken from the [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) repository's samples.

See `models/LICENSE-*` for details.
