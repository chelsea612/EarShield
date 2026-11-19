package app.earshield.core

import org.json.JSONArray
import java.io.File
import java.util.concurrent.Executor
import kotlin.math.log10
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/** Natural covering, noise reduction, chatter reduction and masking noise. */
class SoundTest {
    private val modelsDir = File(System.getProperty("models.dir"))
    private fun resource(name: String) = File(SoundTest::class.java.getResource("/$name")!!.toURI())
    private fun model(name: String) = File(modelsDir, name).readBytes()
    private val encoder by lazy { WespeakerEncoder(model("speaker_encoder.onnx")) }
    private val segmenter by lazy { SpeakerSegmenter(model("speaker_segmentation.onnx")) }

    private fun pipeline(listener: BlockingListener? = null) = BlockingPipeline(
        encoder, segmenter, SileroVad(model("silero_vad.onnx")), listener, null, Executor { it.run() }, Random(1),
        enhancerFactory = { SpeechEnhancer(model("speech_enhancer.onnx")) },
    )

    private fun blockedVoice(): VoiceProfile {
        val p = pipeline()
        val audio = readWav(resource("blocked_enroll.wav"))
        p.startEnrollment()
        val out = FloatArray(AudioFormat.CHUNK)
        for (c in 0 until audio.size / AudioFormat.CHUNK) p.process(audio.copyOfRange(c * AudioFormat.CHUNK, (c + 1) * AudioFormat.CHUNK), out)
        return VoiceProfile("b", "Blocked", Enrollment.voiceprint(encoder, p.finishEnrollment()!!)!!)
    }

    /** Runs [file] through a pipeline and returns the output lined up with the input. */
    private fun run(file: String, settings: BlockingSettings, voices: List<VoiceProfile>): FloatArray {
        val audio = readWav(resource(file))
        val p = pipeline()
        p.settings = settings
        p.voices = voices
        val delay = AudioFormat.samples(settings.delaySeconds)
        val chunks = (audio.size + delay) / AudioFormat.CHUNK + 1
        val output = FloatArray(chunks * AudioFormat.CHUNK)
        val out = FloatArray(AudioFormat.CHUNK)
        for (c in 0 until chunks) {
            val chunk = FloatArray(AudioFormat.CHUNK) { audio.getOrElse(c * AudioFormat.CHUNK + it) { 0f } }
            p.process(chunk, out)
            out.copyInto(output, c * AudioFormat.CHUNK)
        }
        return FloatArray(audio.size) { output.getOrElse(it + delay) { 0f } }
    }

    /**
     * Least-squares fit of [out] ≈ Σ c_k · source_k over [from, to). Returns each source's kept level in dB
     * (0 dB = all of it, −20 dB = a tenth).
     */
    private fun keptDb(out: FloatArray, sources: Map<String, FloatArray>, from: Int, to: Int): Map<String, Double> {
        // Only sources that actually sound in this stretch can be measured.
        val names = sources.keys.filter { k -> (from until to).sumOf { sources[k]!![it].toDouble().let { v -> v * v } } > 1e-6 }
        val n = names.size
        val a = Array(n) { DoubleArray(n) }
        val b = DoubleArray(n)
        for (i in from until to) for (p in 0 until n) {
            val sp = sources[names[p]]!![i].toDouble()
            b[p] += sp * out[i]
            for (q in 0 until n) a[p][q] += sp * sources[names[q]]!![i]
        }
        // Solve a·c = b (Gaussian elimination; n ≤ 3).
        for (c in 0 until n) {
            val piv = (c until n).maxBy { kotlin.math.abs(a[it][c]) }
            a[c] = a[piv].also { a[piv] = a[c] }; b[c] = b[piv].also { b[piv] = b[c] }
            for (r in 0 until n) if (r != c) {
                val f = a[r][c] / a[c][c]
                for (k in 0 until n) a[r][k] -= f * a[c][k]
                b[r] -= f * b[c]
            }
        }
        return names.withIndex().associate { (i, name) -> name to 20 * log10(kotlin.math.abs(b[i] / a[i][i]) + 1e-6) }
    }

    private fun sources(prefix: String, vararg who: String) = who.associateWith { readWav(resource("${prefix}_src_$it.wav")) }

    /** Stretches where only [who] (by the labels) talks. */
    private fun soloRanges(file: String, who: String): List<IntRange> {
        val labels = JSONArray(resource(file.removeSuffix(".wav") + "_labels.json").readText())
        return (0 until labels.length()).map { labels.getJSONObject(it) }.filter { it.getString("who") == who }
            .map { it.getInt("start") until it.getInt("end") }
    }

    private fun report(title: String, out: FloatArray, src: Map<String, FloatArray>, ranges: List<IntRange>): Map<String, Double> {
        val sums = HashMap<String, Double>()
        val counts = HashMap<String, Int>()
        for (r in ranges) keptDb(out, src, r.first, r.last + 1).forEach { (k, v) -> sums.merge(k, v, Double::plus); counts.merge(k, 1, Int::plus) }
        for (k in sums.keys) sums[k] = sums[k]!! / counts[k]!!
        println("    %-34s %s".format(title, sums.entries.sortedBy { it.key }.joinToString { "%s %.1f dB".format(it.key, it.value) }))
        return sums
    }

    @Test
    fun naturalCoverRemovesTheVoiceButKeepsTheRoom() {
        val voice = blockedVoice()
        val src = sources("noisy_turns", "friend", "blocked", "noise")
        val natural = run("noisy_turns.wav", BlockingSettings(cover = CoverStyle.NATURAL), listOf(voice))
        val silence = run("noisy_turns.wav", BlockingSettings(cover = CoverStyle.SILENCE), listOf(voice))
        val blockedTurns = soloRanges("noisy_turns.wav", "blocked")
        val friendTurns = soloRanges("noisy_turns.wav", "friend")
        val n = report("natural, while blocked talks:", natural, src, blockedTurns)
        val sil = report("silence, while blocked talks:", silence, src, blockedTurns)
        val f = report("natural, while friend talks:", natural, src, friendTurns)
        // Room tone isn't the same waveform as the noise at that moment, so compare loudness instead.
        fun levelIn(x: FloatArray, rs: List<IntRange>) = sqrt(rs.sumOf { r -> r.sumOf { x[it].toDouble() * x[it] } } / rs.sumOf { it.count() })
        val noiseLevel = levelIn(src["noise"]!!, blockedTurns)
        val naturalLevel = levelIn(natural, blockedTurns)
        val silenceLevel = levelIn(silence, blockedTurns)
        println("    while blocked talks: noise %.4f, natural output %.4f, silence output %.4f".format(noiseLevel, naturalLevel, silenceLevel))
        assertTrue(n["blocked"]!! < -20, "blocked voice not removed")
        assertTrue(naturalLevel in noiseLevel * 0.5..noiseLevel * 2.0, "room sound not kept at its level while covering")
        assertTrue(silenceLevel < naturalLevel * 0.5, "silence should drop the room sound too")
        assertTrue(f["friend"]!! > -1.5, "friend not kept")
    }

    @Test
    fun noiseReductionRemovesTraffic() {
        val src = sources("noisy_turns", "friend", "blocked", "noise")
        val out = run("noisy_turns.wav", BlockingSettings(noiseReduction = 1f), emptyList())
        val all = listOf(0 until readWav(resource("noisy_turns.wav")).size)
        val r = report("noise reduction at full:", out, src, all)
        assertTrue(r["noise"]!! < -12, "traffic not reduced")
        // The enhancer turns some voices down slightly (about 3 dB for one test speaker) at full strength.
        assertTrue(r["friend"]!! > -4 && r["blocked"]!! > -4, "voices hurt by noise reduction")
    }

    /** Distant chatter is turned down in the pauses between the near conversation's turns. */
    @Test
    fun chatterReductionTurnsDownDistantTalkers() {
        val src = sources("chatter", "A", "C", "far")
        val labels = JSONArray(resource("chatter_labels.json").readText())
        val turns = (0 until labels.length()).map { labels.getJSONObject(it) }.map { it.getInt("start") until it.getInt("end") }
        val pauses = turns.zipWithNext { a, b -> (a.last + 1 + 4000) until (b.first - 4000) }.filter { it.last > it.first }
        val off = run("chatter.wav", BlockingSettings(), emptyList())
        val on = run("chatter.wav", BlockingSettings(chatterReduction = 1f), emptyList())
        for (r in pauses) {
            fun rms(x: FloatArray) = sqrt(r.sumOf { x[it].toDouble() * x[it] } / r.count())
            println("    pause %.2f–%.2f s: output RMS off %.4f, on %.4f".format(r.first / 16000f, r.last / 16000f, rms(off), rms(on)))
        }
        val offP = report("chatter off, in pauses:", off, src, pauses)
        val onP = report("chatter full, in pauses:", on, src, pauses)
        val onT = report("chatter full, while near talk:", on, src, turns)
        // Varies by pause (1–10 dB here): the segmentation model doesn't always notice a faint voice.
        assertTrue(onP["far"]!! < offP["far"]!! - 6, "distant talker not turned down in pauses")
        assertTrue(onT["A"]!! > -1.5 && onT["C"]!! > -1.5, "near conversation hurt")
    }

    @Test
    fun maskingNoiseHasTheRightLevel() {
        val out = run("noisy_turns.wav", BlockingSettings(maskingNoise = 1f, noiseReduction = 1f), emptyList())
        val plain = run("noisy_turns.wav", BlockingSettings(noiseReduction = 1f), emptyList())
        var d = 0.0
        for (i in out.indices) d += (out[i] - plain[i]).toDouble().let { it * it }
        val rms = sqrt(d / out.size)
        println("    masking noise RMS %.4f (target %.4f)".format(rms, BlockingPipeline.MASK_MAX_RMS))
        assertTrue(rms in BlockingPipeline.MASK_MAX_RMS * 0.6..BlockingPipeline.MASK_MAX_RMS * 1.6)
    }
}
