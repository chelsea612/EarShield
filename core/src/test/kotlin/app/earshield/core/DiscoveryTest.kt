package app.earshield.core

import org.json.JSONArray
import java.io.File
import java.util.concurrent.Executor
import kotlin.random.Random
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertTrue

class DiscoveryTest {
    private val modelsDir = File(System.getProperty("models.dir"))
    private fun resource(name: String) = File(DiscoveryTest::class.java.getResource("/$name")!!.toURI())
    private fun model(name: String) = File(modelsDir, name).readBytes()
    private val encoder by lazy { WespeakerEncoder(model("speaker_encoder.onnx")) }
    private val segmenter by lazy { SpeakerSegmenter(model("speaker_segmentation.onnx")) }
    private val separator by lazy { SpeechSeparator(model("speech_separator.onnx")) }

    /** Reference voiceprints for A, B, C from the stretches where each talks alone (per the labels). */
    private fun references(file: String): Map<String, FloatArray> {
        val audio = readWav(resource(file))
        val labels = JSONArray(resource(file.removeSuffix(".wav") + "_labels.json").readText())
        val who = Array(audio.size) { mutableSetOf<String>() }
        for (i in 0 until labels.length()) {
            val l = labels.getJSONObject(i)
            for (n in l.getInt("start") until minOf(l.getInt("end"), audio.size)) who[n] += l.getString("who")
        }
        return listOf("A", "B", "C").associateWith { p ->
            val solo = audio.indices.filter { who[it] == setOf(p) }.map { audio[it] }.toFloatArray()
            Enrollment.voiceprint(encoder, solo)!!
        }
    }

    private fun check(file: String, minSim: Float) {
        val audio = readWav(resource(file))
        val refs = references(file)
        val t0 = System.nanoTime()
        val found = VoiceDiscovery(encoder, segmenter, separator).discover(audio)
        println("$file: found ${found.size} voices in %.1f s of CPU for %.0f s of audio".format(
            (System.nanoTime() - t0) / 1e9, audio.size / 16000f))
        val assigned = found.map { v ->
            val sims = refs.mapValues { Vectors.dot(v.embedding, it.value) }
            val best = sims.maxBy { it.value }
            println("    voice: %.1f s of speech, consistency %.2f, clip %.1f s -> %s".format(
                v.speechSeconds, v.consistency, v.clip.size / 16000f,
                sims.entries.sortedByDescending { it.value }.joinToString { "%s %.2f".format(it.key, it.value) }))
            best.key to best.value
        }
        // Every person is found, as their own voice, with a voiceprint that clearly matches them.
        for (p in listOf("A", "B", "C")) {
            assertTrue(assigned.any { it.first == p && it.second >= minSim }, "$file: no good voice for $p")
        }
        assertTrue(found.size <= 4, "$file: too many voices (${found.size}): people split up")
        for (v in found) assertTrue(v.clip.size >= 16000, "clip too short")
    }

    @Test fun findsEachPersonInAConversation() = check("three_people.wav", 0.6f)

    @Test fun findsEachPersonInANoisyRoom() = check("three_people_room.wav", 0.45f)

    /** Allow only the friend; everyone else (here, the other speaker) is covered. */
    @Test
    fun blockStrangersCoversEveryoneNotAllowed() {
        val friend = VoiceProfile("f", "Friend", Enrollment.voiceprint(encoder, readWav(resource("friend_enroll.wav")))!!, allowed = true)
        val audio = readWav(resource("turns.wav"))
        val p = BlockingPipeline(encoder, segmenter, SileroVad(model("silero_vad.onnx")), null, null, Executor { it.run() }, Random(1))
        p.settings = BlockingSettings(cover = CoverStyle.SILENCE, blockStrangers = true)
        p.voices = listOf(friend)
        val delay = AudioFormat.samples(p.settings.delaySeconds)
        val chunks = (audio.size + delay) / AudioFormat.CHUNK + 1
        val output = FloatArray(chunks * AudioFormat.CHUNK)
        val out = FloatArray(AudioFormat.CHUNK)
        for (c in 0 until chunks) {
            val chunk = FloatArray(AudioFormat.CHUNK) { audio.getOrElse(c * AudioFormat.CHUNK + it) { 0f } }
            p.process(chunk, out)
            out.copyInto(output, c * AudioFormat.CHUNK)
        }
        val labels = JSONArray(resource("turns_labels.json").readText())
        val kept = HashMap<String, DoubleArray>()
        for (i in 0 until labels.length()) {
            val l = labels.getJSONObject(i)
            val e = kept.getOrPut(l.getString("who")) { DoubleArray(2) }
            for (n in l.getInt("start") until minOf(l.getInt("end"), audio.size)) {
                e[0] += audio[n] * audio[n].toDouble()
                val o = output.getOrElse(n + delay) { 0f }
                e[1] += o * o.toDouble()
            }
        }
        val friendKept = sqrt(kept["friend"]!![1] / kept["friend"]!![0])
        val strangerKept = sqrt(kept["blocked"]!![1] / kept["blocked"]!![0])
        println("block strangers: friend %.0f%% kept, stranger %.0f%% kept".format(friendKept * 100, strangerKept * 100))
        assertTrue(friendKept > 0.9, "allowed friend was covered")
        assertTrue(strangerKept < 0.1, "stranger was heard")
    }
}
