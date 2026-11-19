package app.earshield.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.sqrt
import kotlin.math.cos
import kotlin.math.PI

/** Turns speech audio into a voiceprint: a 256-number vector with length 1. */
interface VoiceEncoder {
    fun embed(samples: FloatArray, normalizeVolume: Boolean = true): FloatArray
}

/** Estimates the probability that a 512-sample chunk contains speech. Stateful: feed chunks in order. */
interface VoiceActivityDetector {
    fun speechProbability(chunk: FloatArray): Float
    fun reset()
}

private fun sessionOptions(threads: Int) = OrtSession.SessionOptions().apply {
    setIntraOpNumThreads(threads)
    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
}

/** Silero VAD (v5 or later) running on ONNX Runtime. */
class SileroVad(
    modelBytes: ByteArray,
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment(),
) : VoiceActivityDetector, AutoCloseable {
    private val session = env.createSession(modelBytes, sessionOptions(1))
    private val state = FloatArray(2 * 128)
    private val input = FloatArray(CONTEXT + AudioFormat.CHUNK)
    private val sampleRate = LongBuffer.wrap(longArrayOf(AudioFormat.SAMPLE_RATE.toLong()))

    override fun speechProbability(chunk: FloatArray): Float {
        require(chunk.size == AudioFormat.CHUNK) { "VAD needs ${AudioFormat.CHUNK}-sample chunks" }
        // The model expects the last 64 samples of the previous chunk in front of the new one.
        System.arraycopy(chunk, 0, input, CONTEXT, chunk.size)
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, input.size.toLong()))
        val stateTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(state), longArrayOf(2, 1, 128))
        sampleRate.rewind()
        val srTensor = OnnxTensor.createTensor(env, sampleRate, longArrayOf())
        try {
            session.run(mapOf("input" to inputTensor, "state" to stateTensor, "sr" to srTensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val prob = (result[0].value as Array<FloatArray>)[0][0]
                @Suppress("UNCHECKED_CAST")
                val newState = result[1].value as Array<Array<FloatArray>>
                var i = 0
                for (a in newState) for (b in a) for (v in b) state[i++] = v
                System.arraycopy(input, input.size - CONTEXT, input, 0, CONTEXT)
                return prob
            }
        } finally {
            inputTensor.close(); stateTensor.close(); srTensor.close()
        }
    }

    override fun reset() {
        state.fill(0f)
        input.fill(0f)
    }

    override fun close() = session.close()

    private companion object {
        const val CONTEXT = 64
    }
}

/**
 * WeSpeaker ResNet34 voice encoder (VoxCeleb2-trained, much more robust to noise and echo than GE2E).
 * Input: Kaldi fbank (Hamming window, samples scaled to 16-bit range) with the mean subtracted.
 */
class WespeakerEncoder(
    modelBytes: ByteArray,
    threads: Int = 2,
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment(),
) : VoiceEncoder, AutoCloseable {
    private val session = env.createSession(modelBytes, sessionOptions(threads))
    private val fbank = KaldiFbank(window = KaldiFbank.Window.HAMMING)

    override fun embed(samples: FloatArray, normalizeVolume: Boolean): FloatArray {
        val feats = fbank.compute(samples, scale = 32768f)
        val frames = feats.size / fbank.numBins
        require(frames > 0) { "need at least 25 ms of audio" }
        OnnxTensor.createTensor(env, FloatBuffer.wrap(feats), longArrayOf(1, frames.toLong(), fbank.numBins.toLong()))
            .use { tensor ->
                session.run(mapOf("feats" to tensor)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    return Vectors.normalized((result[0].value as Array<FloatArray>)[0])
                }
            }
    }

    override fun close() = session.close()
}

/**
 * pyannote segmentation-3.0: for a 10 s window, says which of up to 3 local speakers talk in each
 * ~17 ms frame, including overlaps. "Local" means the labels only hold within one window.
 */
class SpeakerSegmenter(
    modelBytes: ByteArray,
    threads: Int = 1,
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment(),
) : AutoCloseable {
    private val session = env.createSession(modelBytes, sessionOptions(threads))

    /** Speaker activity for one window: [speaker][frame]. Frame f covers samples f*N/frames until (f+1)*N/frames. */
    class Activity(val active: Array<BooleanArray>, val windowSamples: Int) {
        val frames: Int get() = active[0].size
        fun frameStart(f: Int): Int = (f.toLong() * windowSamples / frames).toInt()
        fun speakersAt(f: Int): Int = active.count { it[f] }
    }

    fun run(window: FloatArray): Activity {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(window), longArrayOf(1, 1, window.size.toLong())).use { t ->
            session.run(mapOf("x" to t)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val logits = (result[0].value as Array<Array<FloatArray>>)[0] // [frames][7]
                val active = Array(SPEAKERS) { BooleanArray(logits.size) }
                for (f in logits.indices) {
                    var best = 0
                    for (c in 1 until logits[f].size) if (logits[f][c] > logits[f][best]) best = c
                    for (s in POWERSET[best]) active[s][f] = true
                }
                return Activity(active, window.size)
            }
        }
    }

    override fun close() = session.close()

    companion object {
        const val SPEAKERS = 3
        const val WINDOW_SAMPLES = 160_000 // the model's 10 s training window

        /** Output classes: nobody, each speaker alone, each pair together. */
        private val POWERSET = arrayOf(
            intArrayOf(), intArrayOf(0), intArrayOf(1), intArrayOf(2),
            intArrayOf(0, 1), intArrayOf(0, 2), intArrayOf(1, 2),
        )
    }
}

/**
 * Conv-TasNet (Libri2Mix, 16 kHz): splits a mix of two voices into two separate streams.
 * The streams come out at an arbitrary level and in no particular order.
 */
class SpeechSeparator(
    modelBytes: ByteArray,
    threads: Int = 2,
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment(),
) : AutoCloseable {
    private val session = env.createSession(modelBytes, sessionOptions(threads))

    fun separate(mix: FloatArray): Array<FloatArray> {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(mix), longArrayOf(1, mix.size.toLong())).use { t ->
            session.run(mapOf("mixture" to t)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val out = (result[0].value as Array<Array<FloatArray>>)[0]
                return Array(out.size) { out[it].copyOf(mix.size) }
            }
        }
    }

    override fun close() = session.close()
}

/**
 * GTCRN streaming speech enhancement (0.5 MB, trained on DNS3): splits the room's sound into speech and
 * everything else, one 16 ms step at a time. Feed audio in any chunk size. Output sample n is the
 * speech-only version of input sample n − [HOP] (outputs start with [HOP] samples for the padding).
 */
class SpeechEnhancer(
    modelBytes: ByteArray,
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment(),
) : AutoCloseable {
    private val session = env.createSession(modelBytes, sessionOptions(1))
    private val window = FloatArray(N_FFT) { sqrt(0.5 - 0.5 * cos(2 * PI * it / N_FFT)).toFloat() }
    private val fft = Fft(N_FFT)
    private val frame = FloatArray(N_FFT)
    private var filled = HOP // the first frame starts with 256 zeros, like the reference implementation
    private val overlap = FloatArray(N_FFT)
    private val convCache = FloatArray(2 * 1 * 16 * 16 * 33)
    private val traCache = FloatArray(2 * 3 * 1 * 1 * 16)
    private val interCache = FloatArray(2 * 1 * 33 * 16)
    private val re = DoubleArray(N_FFT)
    private val im = DoubleArray(N_FFT)
    private val mix = FloatArray(BINS * 2)

    /**
     * Adds [input] and returns the speech-only samples that became ready (one block of 256 per
     * 256 samples fed, starting once the first 256 have arrived).
     */
    fun process(input: FloatArray): FloatArray {
        val out = ArrayList<FloatArray>()
        for (x in input) {
            frame[filled++] = x
            if (filled == N_FFT) {
                out += step()
                System.arraycopy(frame, HOP, frame, 0, N_FFT - HOP)
                filled = N_FFT - HOP
            }
        }
        val result = FloatArray(out.sumOf { it.size })
        var o = 0
        for (b in out) { b.copyInto(result, o); o += b.size }
        return result
    }

    /** Runs one 512-sample frame and returns the next 256 finished output samples. */
    private fun step(): FloatArray {
        for (i in 0 until N_FFT) { re[i] = (frame[i] * window[i]).toDouble(); im[i] = 0.0 }
        fft.transform(re, im)
        for (k in 0 until BINS) { mix[k * 2] = re[k].toFloat(); mix[k * 2 + 1] = im[k].toFloat() }
        val tensors = listOf(
            OnnxTensor.createTensor(env, FloatBuffer.wrap(mix), longArrayOf(1, BINS.toLong(), 1, 2)),
            OnnxTensor.createTensor(env, FloatBuffer.wrap(convCache), longArrayOf(2, 1, 16, 16, 33)),
            OnnxTensor.createTensor(env, FloatBuffer.wrap(traCache), longArrayOf(2, 3, 1, 1, 16)),
            OnnxTensor.createTensor(env, FloatBuffer.wrap(interCache), longArrayOf(2, 1, 33, 16)),
        )
        try {
            session.run(mapOf("mix" to tensors[0], "conv_cache" to tensors[1], "tra_cache" to tensors[2], "inter_cache" to tensors[3])).use { r ->
                (r[0] as OnnxTensor).floatBuffer.get(mix)
                (r[1] as OnnxTensor).floatBuffer.get(convCache)
                (r[2] as OnnxTensor).floatBuffer.get(traCache)
                (r[3] as OnnxTensor).floatBuffer.get(interCache)
            }
        } finally {
            tensors.forEach { it.close() }
        }
        // Inverse real FFT via the full conjugate-symmetric spectrum.
        for (k in 0 until BINS) { re[k] = mix[k * 2].toDouble(); im[k] = mix[k * 2 + 1].toDouble() }
        for (k in BINS until N_FFT) { re[k] = re[N_FFT - k]; im[k] = -im[N_FFT - k] }
        fft.transform(re, im, inverse = true)
        for (i in 0 until N_FFT) overlap[i] += (re[i] / N_FFT).toFloat() * window[i]
        val ready = overlap.copyOfRange(0, HOP)
        System.arraycopy(overlap, HOP, overlap, 0, N_FFT - HOP)
        overlap.fill(0f, N_FFT - HOP, N_FFT)
        return ready
    }

    override fun close() = session.close()

    companion object {
        const val N_FFT = 512
        const val HOP = 256
        const val BINS = N_FFT / 2 + 1
    }
}
