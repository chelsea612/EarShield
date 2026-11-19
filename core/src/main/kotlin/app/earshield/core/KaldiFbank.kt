package app.earshield.core

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin

/**
 * Log mel filterbank features matching Kaldi's `compute-fbank` (and `kaldi_native_fbank`) defaults:
 * 25 ms frames every 10 ms without edge padding, DC removal, 0.97 pre-emphasis, 512-point FFT,
 * power spectrum, 80 mel bins from 20 Hz to Nyquist, no dither.
 * This is the input that WeSpeaker and 3D-Speaker voice models were trained on.
 */
class KaldiFbank(
    val numBins: Int = 80,
    val window: Window = Window.HAMMING,
    private val sampleRate: Int = AudioFormat.SAMPLE_RATE,
) {
    enum class Window { HAMMING, POVEY }

    private val frameLength = sampleRate * 25 / 1000
    private val frameShift = sampleRate * 10 / 1000
    private val fftSize = 512
    private val windowFn = DoubleArray(frameLength) {
        val a = 2 * PI / (frameLength - 1)
        when (window) {
            Window.HAMMING -> 0.54 - 0.46 * cos(a * it)
            Window.POVEY -> (0.5 - 0.5 * cos(a * it)).pow(0.85)
        }
    }
    private val melBanks: Array<DoubleArray> = buildMelBanks()
    private val melStart = IntArray(numBins) { b -> melBanks[b].indexOfFirst { it > 0 }.coerceAtLeast(0) }
    private val melEnd = IntArray(numBins) { b -> melBanks[b].indexOfLast { it > 0 } + 1 }
    private val cosT = DoubleArray(fftSize / 2) { cos(-2 * PI * it / fftSize) }
    private val sinT = DoubleArray(fftSize / 2) { sin(-2 * PI * it / fftSize) }

    fun frameCount(numSamples: Int): Int = if (numSamples < frameLength) 0 else 1 + (numSamples - frameLength) / frameShift

    /**
     * Returns [frameCount] x [numBins] log-mel values (frame-major), with the per-bin mean over time
     * subtracted when [subtractMean] is true. [scale] multiplies the samples first (WeSpeaker uses 32768).
     */
    fun compute(samples: FloatArray, scale: Float = 1f, subtractMean: Boolean = true): FloatArray {
        val frames = frameCount(samples.size)
        val out = FloatArray(frames * numBins)
        val re = DoubleArray(fftSize)
        val im = DoubleArray(fftSize)
        val frame = DoubleArray(frameLength)
        for (t in 0 until frames) {
            val start = t * frameShift
            var mean = 0.0
            for (i in 0 until frameLength) {
                frame[i] = samples[start + i].toDouble() * scale
                mean += frame[i]
            }
            mean /= frameLength
            for (i in 0 until frameLength) frame[i] -= mean
            for (i in frameLength - 1 downTo 1) frame[i] -= PREEMPH * frame[i - 1]
            frame[0] -= PREEMPH * frame[0]
            re.fill(0.0)
            im.fill(0.0)
            for (i in 0 until frameLength) re[i] = frame[i] * windowFn[i]
            fft(re, im)
            val base = t * numBins
            for (b in 0 until numBins) {
                val w = melBanks[b]
                var e = 0.0
                for (k in melStart[b] until melEnd[b]) e += w[k] * (re[k] * re[k] + im[k] * im[k])
                out[base + b] = ln(max(e, EPSILON)).toFloat()
            }
        }
        if (subtractMean && frames > 0) {
            for (b in 0 until numBins) {
                var m = 0.0
                for (t in 0 until frames) m += out[t * numBins + b]
                val mf = (m / frames).toFloat()
                for (t in 0 until frames) out[t * numBins + b] -= mf
            }
        }
        return out
    }

    /** In-place iterative radix-2 FFT of size [fftSize]. */
    private fun fft(re: DoubleArray, im: DoubleArray) {
        val n = fftSize
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                var tmp = re[i]; re[i] = re[j]; re[j] = tmp
                tmp = im[i]; im[i] = im[j]; im[j] = tmp
            }
        }
        var len = 2
        while (len <= n) {
            val step = n / len
            for (i in 0 until n step len) {
                for (k in 0 until len / 2) {
                    val wr = cosT[k * step]
                    val wi = sinT[k * step]
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

    private fun buildMelBanks(): Array<DoubleArray> {
        fun mel(f: Double) = 1127.0 * ln(1 + f / 700.0)
        val numFftBins = fftSize / 2
        val binWidth = sampleRate.toDouble() / fftSize
        val lowMel = mel(LOW_FREQ)
        val highMel = mel(sampleRate / 2.0)
        val delta = (highMel - lowMel) / (numBins + 1)
        return Array(numBins) { b ->
            val left = lowMel + b * delta
            val center = left + delta
            val right = center + delta
            DoubleArray(numFftBins + 1) { k ->
                if (k >= numFftBins) return@DoubleArray 0.0
                val m = mel(binWidth * k)
                when {
                    m > left && m <= center -> (m - left) / (center - left)
                    m > center && m < right -> (right - m) / (right - center)
                    else -> 0.0
                }
            }
        }
    }

    private companion object {
        const val PREEMPH = 0.97
        const val LOW_FREQ = 20.0
        const val EPSILON = 1.1920928955078125e-07 // FLT_EPSILON, Kaldi's log floor
    }
}
