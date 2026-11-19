package app.earshield.core

import org.json.JSONObject
import java.io.File
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

class EnhancerTest {
    private fun resource(name: String) = File(EnhancerTest::class.java.getResource("/$name")!!.toURI())

    @Test
    fun enhancerMatchesPython() {
        val enh = SpeechEnhancer(File(System.getProperty("models.dir"), "speech_enhancer.onnx").readBytes())
        val audio = readWav(resource("noisy_turns.wav")).copyOf(2 * 16000 + 512)
        val ref = JSONObject(resource("reference_enhance.json").readText()).getJSONArray("enhanced")
        // Feed in app-sized chunks. The first 256 outputs belong to the zero padding before the audio,
        // so output n is the speech-only version of input n − 256.
        val out = ArrayList<Float>()
        for (c in 0 until audio.size / 512) out += enh.process(audio.copyOfRange(c * 512, (c + 1) * 512)).toList()
        var maxErr = 0f
        var peak = 0f
        var worst = 0
        // The reference padded the end of its 2 s clip with silence; skip the last frame, which sees it.
        for (i in 0 until ref.length() - SpeechEnhancer.N_FFT) {
            val r = ref.getDouble(i).toFloat()
            val e = abs(out[i + SpeechEnhancer.HOP] - r)
            if (e > maxErr) { maxErr = e; worst = i }
            peak = maxOf(peak, abs(r))
        }
        println("enhancer: max error %.2e at sample %d (signal peak %.3f)".format(maxErr, worst, peak))
        assertTrue(maxErr < peak * 1e-3f, "enhancer differs from Python")
    }
}
