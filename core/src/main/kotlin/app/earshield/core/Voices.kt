package app.earshield.core

/**
 * A voiceprint. Only the voiceprint is kept, never audio.
 * Blocked voices get covered. Allowed voices (you, or "not them" corrections) are never covered:
 * a voice is only blocked if it sounds clearly more like a blocked voice than any allowed one.
 */
class VoiceProfile(
    val id: String,
    val name: String,
    val embedding: FloatArray,
    val createdAtMillis: Long = System.currentTimeMillis(),
    val allowed: Boolean = false,
)

/** [voice] is the closest blocked voice; [allowedScore] is the similarity to the closest allowed voice. */
data class VoiceMatch(val voice: VoiceProfile?, val score: Float, val allowedScore: Float = 0f)

object VoiceMatcher {
    /** The blocked voice most similar to [embedding] (cosine similarity, −1 to 1), plus the closest allowed score. */
    fun best(embedding: FloatArray, voices: List<VoiceProfile>): VoiceMatch {
        var best: VoiceProfile? = null
        var bestScore = -1f
        var allowedScore = 0f
        for (v in voices) {
            val s = Vectors.dot(embedding, v.embedding)
            if (v.allowed) {
                allowedScore = maxOf(allowedScore, s)
            } else if (s > bestScore) {
                bestScore = s
                best = v
            }
        }
        return VoiceMatch(best, if (best == null) 0f else bestScore, allowedScore)
    }
}

object Enrollment {
    /** Minimum speech needed for a usable voiceprint. */
    const val MIN_SECONDS = 6f

    /** Enrollment stops collecting after this much speech. */
    const val MAX_SECONDS = 20f

    /** Voiceprints are averaged over 3 s slices, the same length the pipeline checks with. */
    private const val PARTIAL_SECONDS = 3f

    /**
     * Builds a voiceprint from speech-only audio by averaging 3 s slices with 50% overlap.
     * Returns null if there is less than [MIN_SECONDS] of speech.
     */
    fun voiceprint(encoder: VoiceEncoder, speech: FloatArray): FloatArray? = analyze(encoder, speech)?.embedding

    /**
     * A voiceprint plus [consistency]: how alike its slices sound (average similarity of each slice
     * to the whole). One clear voice scores high; a recording that mixes voices or is mostly noise
     * scores low and makes a voiceprint that matches everyone.
     */
    class Result(val embedding: FloatArray, val consistency: Float) {
        val looksMixed: Boolean get() = consistency < MIN_CONSISTENCY
    }

    /** Below this, a recording probably has more than one voice in it (one voice in a living room: >= 0.75; two: <= 0.73). */
    const val MIN_CONSISTENCY = 0.74f

    fun analyze(encoder: VoiceEncoder, speech: FloatArray): Result? {
        if (AudioFormat.seconds(speech.size.toLong()) < MIN_SECONDS) return null
        val len = AudioFormat.samples(PARTIAL_SECONDS)
        val step = len / 2
        val parts = (0..speech.size - len step step).map { start ->
            encoder.embed(speech.copyOfRange(start, start + len))
        }
        val center = Vectors.normalized(Vectors.mean(parts))
        val consistency = parts.map { Vectors.dot(it, center) }.average().toFloat()
        return Result(center, consistency)
    }
}
