package app.earshield.core

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executor
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

private val modelsDir = File(System.getProperty("models.dir"))
private fun resource(name: String) = File(CoreTest::class.java.getResource("/$name")!!.toURI())

/** Reads a 16-bit mono WAV file into floats in [-1, 1]. */
fun readWav(file: File): FloatArray {
    val bytes = file.readBytes()
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    var pos = 12
    while (pos + 8 <= bytes.size) {
        val id = String(bytes, pos, 4)
        val size = buf.getInt(pos + 4)
        if (id == "data") return FloatArray(size / 2) { buf.getShort(pos + 8 + it * 2) / 32768f }
        pos += 8 + size
    }
    error("no data chunk in $file")
}

private fun JSONArray.toFloats() = FloatArray(length()) { getDouble(it).toFloat() }

class CoreTest {
    private val ref1 = JSONObject(resource("reference.json").readText())
    private val ref2 = JSONObject(resource("reference_v2.json").readText())
    private val encoder by lazy { WespeakerEncoder(File(modelsDir, "speaker_encoder.onnx").readBytes()) }
    private val segmenter by lazy { SpeakerSegmenter(File(modelsDir, "speaker_segmentation.onnx").readBytes()) }
    private fun newVad() = SileroVad(File(modelsDir, "silero_vad.onnx").readBytes())
    private val separator by lazy { SpeechSeparator(File(modelsDir, "speech_separator.onnx").readBytes()) }

    /** Runs analysis inline so tests are deterministic (the app uses a background thread). */
    private val inline = Executor { it.run() }

    // --- Each component matches its Python reference --------------------------------------------

    @Test
    fun fbankMatchesKaldi() {
        val clip = readWav(resource("blocked_enroll.wav")).copyOf(3 * AudioFormat.SAMPLE_RATE)
        val ours = KaldiFbank().compute(clip, scale = 32768f, subtractMean = false)
        val ref = ref2.getJSONArray("fbank_raw")
        assertTrue(ours.size == ref.length() * 80, "frames ${ours.size / 80} vs ${ref.length()}")
        var maxErr = 0f
        for (t in 0 until ref.length()) {
            val row = ref.getJSONArray(t).toFloats()
            for (b in row.indices) maxErr = maxOf(maxErr, abs(row[b] - ours[t * 80 + b]))
        }
        println("fbank: max log-energy error vs kaldi_native_fbank = $maxErr")
        assertTrue(maxErr < 0.01f, "fbank error $maxErr")
    }

    @Test
    fun encoderMatchesPython() {
        val clip = readWav(resource("blocked_enroll.wav")).copyOf(3 * AudioFormat.SAMPLE_RATE)
        val sim = Vectors.dot(encoder.embed(clip), ref2.getJSONArray("resnet_embedding").toFloats())
        println("encoder: similarity to Python output = $sim")
        assertTrue(sim > 0.9999f, "similarity $sim")
    }

    @Test
    fun segmenterMatchesPython() {
        val audio = readWav(resource("overlap.wav")).copyOf(SpeakerSegmenter.WINDOW_SAMPLES)
        val act = segmenter.run(audio)
        val ref = ref2.getJSONArray("seg_argmax")
        val classes = arrayOf(setOf(), setOf(0), setOf(1), setOf(2), setOf(0, 1), setOf(0, 2), setOf(1, 2))
        var agree = 0
        for (f in 0 until act.frames) {
            val ours = (0 until 3).filter { act.active[it][f] }.toSet()
            if (ours == classes[ref.getInt(f)]) agree++
        }
        println("segmenter: ${agree * 100 / act.frames}% of frames match Python")
        assertTrue(agree >= act.frames * 99 / 100)
    }

    @Test
    fun vadMatchesPython() {
        val audio = readWav(resource("conversation.wav"))
        val ref = ref1.getJSONArray("vad_probs").toFloats()
        newVad().use { vad ->
            var maxErr = 0f
            for (i in ref.indices) {
                val p = vad.speechProbability(audio.copyOfRange(i * 512, i * 512 + 512))
                maxErr = maxOf(maxErr, abs(p - ref[i]))
            }
            assertTrue(maxErr < 1e-3, "vad error $maxErr")
        }
    }

    // --- The whole pipeline on conversations ----------------------------------------------------

    private class Recorder : BlockingListener {
        val events = mutableListOf<String>()
        val heard = mutableListOf<Pair<Long, FloatArray>>()
        var now = 0L
        override fun onBlockStarted(voice: VoiceProfile, score: Float, leakedSeconds: Float, heard: FloatArray) {
            events += "block %.1fs score=%.2f leaked=%.1fs".format(now / 16000f, score, leakedSeconds)
            this.heard += now to heard
        }
        override fun onFellBehind() { events += "fell behind" }
    }

    /** Enrolls a voice the way the app does: speech collected through the pipeline's speech detector. */
    private fun enroll(file: String, name: String = "Blocked person"): VoiceProfile {
        val audio = readWav(resource(file))
        val p = BlockingPipeline(encoder, segmenter, newVad(), analysisExecutor = inline)
        p.startEnrollment()
        val out = FloatArray(AudioFormat.CHUNK)
        for (i in 0 until audio.size / AudioFormat.CHUNK) {
            p.process(audio.copyOfRange(i * AudioFormat.CHUNK, (i + 1) * AudioFormat.CHUNK), out)
        }
        val result = Enrollment.analyze(encoder, p.finishEnrollment()!!)!!
        println("enrolled from $file: consistency %.2f".format(result.consistency))
        return VoiceProfile("b", name, result.embedding)
    }

    private fun run(
        file: String,
        settings: BlockingSettings,
        voices: List<VoiceProfile>,
        separate: Boolean = false,
    ): Pair<FloatArray, Recorder> {
        val audio = readWav(resource(file))
        val rec = Recorder()
        val p = BlockingPipeline(encoder, segmenter, newVad(), rec, if (separate) separator else null, inline, Random(1))
        p.settings = settings
        p.voices = voices
        val delay = AudioFormat.samples(settings.delaySeconds)
        val chunks = (audio.size + delay) / AudioFormat.CHUNK + 1
        val output = FloatArray(chunks * AudioFormat.CHUNK)
        val chunk = FloatArray(AudioFormat.CHUNK)
        val out = FloatArray(AudioFormat.CHUNK)
        val t0 = System.nanoTime()
        for (c in 0 until chunks) {
            for (i in chunk.indices) chunk[i] = audio.getOrElse(c * AudioFormat.CHUNK + i) { 0f }
            rec.now = c.toLong() * AudioFormat.CHUNK
            p.process(chunk, out)
            out.copyInto(output, c * AudioFormat.CHUNK)
        }
        println("$file: %.1f s of CPU for %.1f s of audio; events %s".format(
            (System.nanoTime() - t0) / 1e9, audio.size / 16000f, rec.events))
        return FloatArray(audio.size) { output.getOrElse(it + delay) { 0f } } to rec
    }

    /** Fraction of voice level kept in each kind of region: who is talking alone, or both together. */
    private fun regions(file: String, aligned: FloatArray): Map<String, Double> {
        val audio = readWav(resource(file))
        val labels = JSONArray(resource(file.removeSuffix(".wav") + "_labels.json").readText())
        val who = Array(audio.size) { mutableSetOf<String>() }
        for (i in 0 until labels.length()) {
            val l = labels.getJSONObject(i)
            for (s in l.getInt("start") until minOf(l.getInt("end"), audio.size)) who[s] += l.getString("who")
        }
        val eIn = HashMap<String, Double>()
        val eOut = HashMap<String, Double>()
        for (s in audio.indices) {
            val key = when {
                who[s].size > 1 -> "both talking"
                who[s].isEmpty() || "gap" in who[s] -> continue
                else -> "${who[s].first()} alone"
            }
            eIn.merge(key, audio[s].toDouble() * audio[s], Double::plus)
            eOut.merge(key, aligned[s].toDouble() * aligned[s], Double::plus)
        }
        val kept = eIn.keys.associateWith { sqrt(eOut[it]!! / eIn[it]!!) }
        kept.toSortedMap().forEach { (k, v) -> println("    %-14s %3.0f%% of the voice level kept".format(k, v * 100)) }
        return kept
    }

    private val silence = BlockingSettings(cover = CoverStyle.SILENCE)

    @Test
    fun blocksWithPausesBetweenTurns() {
        val voice = enroll("blocked_enroll.wav")
        val (out, _) = run("conversation.wav", silence, listOf(voice))
        val kept = regions("conversation.wav", out)
        assertTrue(kept["blocked alone"]!! < 0.05, "blocked person heard")
        assertTrue(kept["friend alone"]!! > 0.95, "friend cut")
    }

    @Test
    fun blocksBackToBackTurns() {
        val voice = enroll("blocked_enroll.wav")
        val (out, _) = run("turns.wav", silence, listOf(voice))
        val kept = regions("turns.wav", out)
        assertTrue(kept["blocked alone"]!! < 0.10, "blocked person heard")
        assertTrue(kept["friend alone"]!! > 0.90, "friend cut")
    }

    @Test
    fun turnsDownOverlapsAndKeepsTheFriend() {
        val voice = enroll("blocked_enroll.wav")
        val (out, _) = run("overlap.wav", silence, listOf(voice))
        val inp = readWav(resource("overlap.wav"))
        println("    kept per 0.5 s: " + (0 until inp.size / 8000).joinToString(" ") { b ->
            var a = 0.0; var o = 0.0
            for (i in b * 8000 until (b + 1) * 8000) { a += inp[i] * inp[i].toDouble(); o += out[i] * out[i].toDouble() }
            "%.1f:%d".format(b / 2f, (sqrt(o / a.coerceAtLeast(1e-12)) * 100).toInt())
        })
        val kept = regions("overlap.wav", out)
        assertTrue(kept["blocked alone"]!! < 0.10, "blocked person heard")
        assertTrue(kept["friend alone"]!! > 0.90, "friend cut")
        // The segmentation model marks only about half of each overlap as two people, so overlaps are
        // partly turned down rather than fully. Removing the voice from the mix is stage 2.
        assertTrue(kept["both talking"]!! in 0.10..0.75, "overlap should be turned down, not cut or kept")
    }

    /** Voice level kept in each labeled turn of [file]. */
    private fun turns(file: String, aligned: FloatArray): List<Pair<String, Double>> {
        val audio = readWav(resource(file))
        val labels = JSONArray(resource(file.removeSuffix(".wav") + "_labels.json").readText())
        return (0 until labels.length()).map { labels.getJSONObject(it) }.filter { it.getString("who") != "gap" }.map { l ->
            var eIn = 0.0
            var eOut = 0.0
            for (i in l.getInt("start") until minOf(l.getInt("end"), audio.size)) {
                eIn += audio[i].toDouble() * audio[i]; eOut += aligned[i].toDouble() * aligned[i]
            }
            val kept = sqrt(eOut / eIn)
            println("    %-8s %5.1f–%5.1f s: %3.0f%% kept".format(l.getString("who"), l.getInt("start") / 16000f, l.getInt("end") / 16000f, kept * 100))
            l.getString("who") to kept
        }
    }

    /**
     * In the simulated room the blocked person's first turn is buried in background chatter: even
     * offline it only scores 0.10–0.40 against their voiceprint, so no setting could catch it without
     * also blocking others. Their second turn scores 0.62–0.65. The test requires that recognizable
     * turns are covered and that the friend is never cut.
     */
    @Test
    fun worksInANoisyEchoeyRoom() {
        val voice = enroll("blocked_enroll_room.wav")
        val (out, _) = run("turns_room.wav", silence, listOf(voice))
        val t = turns("turns_room.wav", out)
        val blockedTurns = t.filter { it.first == "blocked" }
        assertTrue(blockedTurns[1].second < 0.15, "recognizable blocked turn heard")
        assertTrue(t.filter { it.first == "friend" }.all { it.second > 0.85 }, "friend cut")
        val (out2, _) = run("overlap_room.wav", silence, listOf(voice))
        val kept = regions("overlap_room.wav", out2)
        assertTrue(kept["friend alone"]!! > 0.85, "friend cut in overlap_room")
    }

    @Test
    fun notThemCorrectionsStopFalsePositives() {
        val voice = enroll("blocked_enroll.wav")
        val loose = silence.copy(threshold = 0.05f) // absurdly low, so the friend gets blocked too
        val (before, rec) = run("turns.wav", loose, listOf(voice))
        println("before corrections:")
        assertTrue(regions("turns.wav", before)["friend alone"]!! < 0.9, "test needs false positives first")
        // "Not them" on blocks that started while the friend had the floor in the first turn (0–4 s, decided by ~5 s).
        val corrections = rec.heard.filter { it.first < 5 * 16000 }.mapIndexed { i, (_, e) ->
            VoiceProfile("c$i", "Not them", e, allowed = true)
        }
        println("corrections: ${corrections.size}")
        assertTrue(corrections.isNotEmpty())
        val (after, _) = run("turns.wav", loose, listOf(voice) + corrections)
        println("after corrections:")
        val kept = regions("turns.wav", after)
        assertTrue(kept["friend alone"]!! > 0.9, "friend still cut")
        assertTrue(kept["blocked alone"]!! < 0.15, "blocked person heard after corrections")
    }

    @Test
    fun recordingCheckSpotsMixedVoices() {
        val single = Enrollment.analyze(encoder, readWav(resource("blocked_enroll.wav")))!!
        val mixed = Enrollment.analyze(encoder, readWav(resource("turns.wav")).copyOf(16000 * 14))!!
        println("consistency: one voice %.3f, two voices %.3f".format(single.consistency, mixed.consistency))
        assertTrue(!single.looksMixed, "single voice flagged as mixed")
        assertTrue(mixed.looksMixed, "mixed recording not flagged")
    }

    /**
     * Least-squares fit of [out] ≈ a·friend + b·blocked over [from, to): returns the blocked person's
     * level relative to the friend in dB, and how much of the friend's level is kept.
     */
    private fun voiceLevels(out: FloatArray, from: Int, to: Int): Pair<Double, Double> {
        val f = readWav(resource("overlap_src_friend.wav"))
        val b = readWav(resource("overlap_src_blocked.wav"))
        var ff = 0.0; var fb = 0.0; var bb = 0.0; var fo = 0.0; var bo = 0.0
        for (i in from until to) {
            ff += f[i] * f[i].toDouble(); fb += f[i] * b[i].toDouble(); bb += b[i] * b[i].toDouble()
            fo += f[i] * out[i].toDouble(); bo += b[i] * out[i].toDouble()
        }
        val det = ff * bb - fb * fb
        val cf = (fo * bb - bo * fb) / det
        val cb = (ff * bo - fb * fo) / det
        val rel = 20 * kotlin.math.log10(abs(cb) * sqrt(bb) / (abs(cf) * sqrt(ff)) + 1e-9)
        return rel to abs(cf)
    }

    @Test
    fun separationRemovesTheBlockedVoiceFromOverlaps() {
        val voice = enroll("blocked_enroll.wav")
        val mix = readWav(resource("overlap.wav"))
        val (plain, _) = run("overlap.wav", silence, listOf(voice))
        val (separated, _) = run("overlap.wav", silence, listOf(voice), separate = true)
        val from = 4 * 16000; val to = 8 * 16000 // both talking
        val (inRel, _) = voiceLevels(mix, from, to)
        val (plainRel, plainFriend) = voiceLevels(plain, from, to)
        val (sepRel, sepFriend) = voiceLevels(separated, from, to)
        println("    while both talk: blocked vs friend %.1f dB in the room; %.1f dB turned down (friend at %.0f%%); %.1f dB separated (friend at %.0f%%)"
            .format(inRel, plainRel, plainFriend * 100, sepRel, sepFriend * 100))
        val kept = regions("overlap.wav", separated)
        assertTrue(sepRel < -10, "blocked voice still prominent during overlap: $sepRel dB")
        assertTrue(sepFriend > 0.6, "friend lost during overlap: ${sepFriend * 100}%")
        assertTrue(kept["blocked alone"]!! < 0.15, "blocked person heard when alone")
        assertTrue(kept["friend alone"]!! > 0.9, "friend cut when alone")
    }

    @Test
    fun separationDoesNotHurtTurnTaking() {
        val voice = enroll("blocked_enroll.wav")
        for (file in listOf("turns.wav", "conversation.wav")) {
            val (out, _) = run(file, silence, listOf(voice), separate = true)
            val kept = regions(file, out)
            assertTrue(kept["blocked alone"]!! < 0.15, "$file: blocked person heard")
            assertTrue(kept["friend alone"]!! > 0.9, "$file: friend cut")
        }
    }

    @Test
    fun passesEverythingWithNoBlockedVoices() {
        val (out, _) = run("overlap.wav", silence, emptyList())
        for ((_, kept) in regions("overlap.wav", out)) assertTrue(kept > 0.99)
    }
}
