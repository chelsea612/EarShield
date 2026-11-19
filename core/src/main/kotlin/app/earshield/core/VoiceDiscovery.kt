package app.earshield.core

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Finds the different people talking in a recorded conversation, so their voices can be blocked or
 * allowed without needing each person alone.
 *
 *  1. Speaker segmentation runs over the recording in 10 s windows (5 s apart) to find who talks when.
 *  2. Each person's solo speech is cut into pieces of up to 3 s, and each piece becomes a voiceprint.
 *  3. Where two people overlap, the separator splits the voices and each half becomes a voiceprint too.
 *  4. Voiceprints that sound alike are grouped (average-linkage clustering); each group is one person.
 */
class VoiceDiscovery(
    private val encoder: VoiceEncoder,
    private val segmenter: SpeakerSegmenter,
    private val separator: SpeechSeparator? = null,
) {
    class FoundVoice(
        /** The person's voiceprint. */
        val embedding: FloatArray,
        /** How much of their speech was found, in seconds. */
        val speechSeconds: Float,
        /** How alike the pieces in this group sound (see [Enrollment.Result.consistency]). */
        val consistency: Float,
        /** A short sample of them talking alone (or separated), for you to listen to. */
        val clip: FloatArray,
    )

    private class Piece(val embedding: FloatArray, val audio: FloatArray, val solo: Boolean) {
        val seconds: Float get() = audio.size / AudioFormat.SAMPLE_RATE.toFloat()
        val weight: Float get() = seconds * (if (solo) 1f else SEPARATED_WEIGHT)
    }

    /** [progress] receives 0–1 as the work goes. Returns the people found, most talkative first. */
    fun discover(audio: FloatArray, progress: (Float) -> Unit = {}): List<FoundVoice> {
        val window = SpeakerSegmenter.WINDOW_SAMPLES
        val step = window / 2
        val starts = if (audio.size <= window) listOf(0) else (0..audio.size - window step step).toMutableList().also {
            if (it.last() + window < audio.size) it += audio.size - window
        }
        val pieces = ArrayList<Piece>()
        for ((w, start) in starts.withIndex()) {
            val chunk = FloatArray(window) { audio.getOrElse(start + it) { 0f } }
            val act = segmenter.run(chunk)
            // Each window "owns" its middle half, so overlapping windows don't count speech twice.
            val ownFrom = if (w == 0) 0 else start + window / 4
            val ownTo = if (w == starts.lastIndex) audio.size else start + window * 3 / 4
            fun owned(s: Int) = s + start in ownFrom until ownTo
            for (k in 0 until SpeakerSegmenter.SPEAKERS) addSoloPieces(chunk, act, k, ::owned, pieces)
            if (separator != null) addSeparatedPieces(chunk, act, ::owned, pieces)
            progress((w + 1f) / starts.size * 0.9f)
        }
        val clusters = cluster(pieces)
        progress(1f)
        return clusters.mapNotNull { members ->
            val seconds = members.sumOf { it.seconds.toDouble() }.toFloat()
            if (seconds < MIN_VOICE_SECONDS) return@mapNotNull null
            val sum = FloatArray(members[0].embedding.size)
            for (p in members) for (i in sum.indices) sum[i] += p.embedding[i] * p.weight
            val center = Vectors.normalized(sum)
            val consistency = members.map { Vectors.dot(it.embedding, center) }.average().toFloat()
            val best = members.filter { it.solo }.maxByOrNull { it.audio.size } ?: members.maxBy { it.audio.size }
            FoundVoice(center, seconds, consistency, best.audio.copyOf(min(best.audio.size, AudioFormat.samples(CLIP_SECONDS))))
        }.sortedByDescending { it.speechSeconds }
    }

    /** Cuts speaker [k]'s solo speech in this window into pieces of up to 3 s and fingerprints each. */
    private fun addSoloPieces(
        chunk: FloatArray,
        act: SpeakerSegmenter.Activity,
        k: Int,
        owned: (Int) -> Boolean,
        out: MutableList<Piece>,
    ) {
        val buf = ArrayList<Float>()
        fun flush() {
            if (buf.size >= AudioFormat.samples(MIN_PIECE_SECONDS)) {
                val a = buf.toFloatArray()
                out += Piece(encoder.embed(a), a, solo = true)
            }
            buf.clear()
        }
        for (f in 0 until act.frames) {
            val solo = act.active[k][f] && act.speakersAt(f) == 1
            val from = act.frameStart(f)
            val to = if (f + 1 < act.frames) act.frameStart(f + 1) else chunk.size
            if (solo && owned((from + to) / 2)) {
                for (i in from until to) buf += chunk[i]
                if (buf.size >= AudioFormat.samples(PIECE_SECONDS)) flush()
            } else if (!solo && buf.isNotEmpty() && f > 0 && act.active[k][f - 1]) {
                // A short gap inside the same person's speech is fine; a change of speaker ends the piece.
                if (act.speakersAt(f) > 0) flush()
            }
        }
        flush()
    }

    /** Separates stretches where exactly two people talk, and fingerprints each separated voice. */
    private fun addSeparatedPieces(
        chunk: FloatArray,
        act: SpeakerSegmenter.Activity,
        owned: (Int) -> Boolean,
        out: MutableList<Piece>,
    ) {
        val sep = separator ?: return
        var f = 0
        while (f < act.frames) {
            if (act.speakersAt(f) == 2 && owned(act.frameStart(f))) {
                var g = f
                while (g < act.frames && act.speakersAt(g) >= 2) g++
                val from = act.frameStart(f)
                val to = if (g < act.frames) act.frameStart(g) else chunk.size
                if (to - from >= AudioFormat.samples(MIN_OVERLAP_SECONDS)) {
                    val ctx = AudioFormat.samples(0.5f)
                    val a = max(0, from - ctx)
                    val b = min(chunk.size, to + ctx)
                    val mix = chunk.copyOfRange(a, b)
                    val streams = sep.separate(mix)
                    val mixRms = rms(mix).coerceAtLeast(1e-9f)
                    for (s in streams) {
                        if (rms(s) / mixRms < 0.1f) continue
                        val scaled = s.copyOf()
                        val gain = mixRms / rms(s).coerceAtLeast(1e-9f) * 0.7f
                        for (i in scaled.indices) scaled[i] *= gain
                        out += Piece(encoder.embed(scaled), scaled, solo = false)
                    }
                }
                f = g
            } else {
                f++
            }
        }
    }

    /** Average-linkage clustering: keep merging the two most alike groups while they're alike enough. */
    private fun cluster(pieces: List<Piece>): List<List<Piece>> {
        val groups = pieces.map { mutableListOf(it) }.toMutableList()
        if (groups.size < 2) return groups
        val n = pieces.size
        val sim = Array(n) { i -> FloatArray(n) { j -> Vectors.dot(pieces[i].embedding, pieces[j].embedding) } }
        val index = HashMap<Piece, Int>().apply { pieces.forEachIndexed { i, p -> put(p, i) } }
        fun linkage(a: List<Piece>, b: List<Piece>): Float {
            var s = 0f
            var w = 0f
            for (p in a) for (q in b) {
                val ww = p.weight * q.weight
                s += sim[index[p]!!][index[q]!!] * ww
                w += ww
            }
            return s / w
        }
        while (groups.size > 1) {
            var best = -2f
            var bi = -1
            var bj = -1
            for (i in groups.indices) for (j in i + 1 until groups.size) {
                val l = linkage(groups[i], groups[j])
                if (l > best) { best = l; bi = i; bj = j }
            }
            if (best < MERGE_THRESHOLD) break
            groups[bi].addAll(groups[bj])
            groups.removeAt(bj)
        }
        return groups
    }

    private fun rms(x: FloatArray): Float {
        var s = 0.0
        for (v in x) s += v * v
        return sqrt(s / x.size.coerceAtLeast(1)).toFloat()
    }

    companion object {
        const val PIECE_SECONDS = 3f
        const val MIN_PIECE_SECONDS = 1.5f
        const val MIN_OVERLAP_SECONDS = 1.0f
        const val SEPARATED_WEIGHT = 0.5f
        const val MERGE_THRESHOLD = 0.40f
        const val MIN_VOICE_SECONDS = 2.5f
        const val CLIP_SECONDS = 4f
        /** Recordings shorter than this rarely give a reliable voice. */
        const val MIN_RECORDING_SECONDS = 15f
        const val MAX_RECORDING_SECONDS = 90f
    }
}
