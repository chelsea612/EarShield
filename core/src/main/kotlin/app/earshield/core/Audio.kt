package app.earshield.core

import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sqrt

object AudioFormat {
    /** All processing runs at 16 kHz mono, which is also what Bluetooth headset mics deliver. */
    const val SAMPLE_RATE = 16_000

    /** Samples per processing chunk (32 ms). Required by the Silero VAD model. */
    const val CHUNK = 512

    fun seconds(samples: Long): Float = samples.toFloat() / SAMPLE_RATE
    fun samples(seconds: Float): Int = (seconds * SAMPLE_RATE).toInt()
}

object Vectors {
    fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }

    fun normalized(v: FloatArray): FloatArray {
        val n = sqrt(dot(v, v))
        return if (n > 0f) FloatArray(v.size) { v[it] / n } else v.copyOf()
    }

    fun mean(vectors: List<FloatArray>): FloatArray {
        val out = FloatArray(vectors.first().size)
        for (v in vectors) for (i in out.indices) out[i] += v[i]
        for (i in out.indices) out[i] /= vectors.size
        return out
    }

    /**
     * Raises quiet audio to an average level of [targetDbfs], as the voice encoder expects.
     * Loud audio is left alone, matching the training preprocessing.
     */
    fun normalizeVolume(samples: FloatArray, targetDbfs: Float = -30f): FloatArray {
        var sumSq = 0.0
        for (s in samples) sumSq += s * s
        val meanSq = sumSq / samples.size.coerceAtLeast(1)
        if (meanSq <= 1e-12) return samples.copyOf()
        val gainDb = targetDbfs - 10 * log10(meanSq)
        if (gainDb <= 0) return samples.copyOf()
        val gain = 10.0.pow(gainDb / 20).toFloat()
        return FloatArray(samples.size) { samples[it] * gain }
    }
}

/** Fixed-size ring of the most recent samples. */
class SampleRing(val capacity: Int) {
    private val data = FloatArray(capacity)
    private var write = 0

    /** Total samples ever written. */
    var written = 0L
        private set

    val size: Int get() = minOf(written, capacity.toLong()).toInt()

    fun write(samples: FloatArray, offset: Int = 0, length: Int = samples.size) {
        for (i in 0 until length) {
            data[write] = samples[offset + i]
            write = (write + 1) % capacity
        }
        written += length
    }

    /** The last [count] samples written, oldest first. */
    fun latest(count: Int): FloatArray {
        val n = minOf(count, size)
        val out = FloatArray(n)
        var r = (write - n + capacity) % capacity
        for (i in 0 until n) {
            out[i] = data[r]
            r = (r + 1) % capacity
        }
        return out
    }

    /** Sample at absolute position [index] (0 = first sample ever written), or 0 if not held. */
    fun at(index: Long): Float {
        if (index < 0 || index >= written || index < written - capacity) return 0f
        return data[(index % capacity).toInt()]
    }

    fun clear() {
        write = 0
        written = 0
    }
}

/** In-place radix-2 complex FFT (forward, or inverse without the 1/N scaling). */
class Fft(private val n: Int) {
    private val cosT = DoubleArray(n / 2) { kotlin.math.cos(-2 * Math.PI * it / n) }
    private val sinT = DoubleArray(n / 2) { kotlin.math.sin(-2 * Math.PI * it / n) }

    init {
        require(n > 0 && n and (n - 1) == 0) { "FFT size must be a power of two" }
    }

    fun transform(re: DoubleArray, im: DoubleArray, inverse: Boolean = false) {
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
        }
        val sign = if (inverse) -1.0 else 1.0
        var len = 2
        while (len <= n) {
            val step = n / len
            for (i in 0 until n step len) {
                for (k in 0 until len / 2) {
                    val wr = cosT[k * step]
                    val wi = sign * sinT[k * step]
                    val a = i + k
                    val b = a + len / 2
                    val xr = re[b] * wr - im[b] * wi
                    val xi = re[b] * wi + im[b] * wr
                    re[b] = re[a] - xr; im[b] = im[a] - xi
                    re[a] += xr; im[a] += xi
                }
            }
            len = len shl 1
        }
    }
}
