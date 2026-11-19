package app.earshield.core

import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

enum class CoverStyle { SILENCE, SOFT_NOISE, QUIETER }

/** What to do when a blocked person talks at the same time as someone else. */
enum class OverlapStyle {
    /** Split the voices and keep the other person (needs a separator; falls back to turning down). */
    SEPARATE,
    TURN_DOWN,
    COVER,
}

data class BlockingSettings(
    /** How far behind real time you hear the room. The app needs ~2 s or more to decide well. */
    val delaySeconds: Float = 3f,
    /** Voice similarity needed to block (cosine, −1 to 1). */
    val threshold: Float = DEFAULT_THRESHOLD,
    val cover: CoverStyle = CoverStyle.SOFT_NOISE,
    val overlap: OverlapStyle = OverlapStyle.SEPARATE,
    /** When false, everything is passed through (blocked voices are still detected and reported). */
    val blockingEnabled: Boolean = true,
    /** Cover everyone who isn't an allowed voice (you, friends, "Not them"), not just blocked voices. */
    val blockStrangers: Boolean = false,
) {
    companion object {
        const val DEFAULT_THRESHOLD = 0.42f
    }
}

/** Someone the app is hearing, for the live "Who's talking" view. */
class LiveSpeaker(
    /** Stable for as long as the app keeps following this voice. */
    val trackId: Int,
    /** The voiceprint built from everything they've said so far (can be saved as a new voice). */
    val embedding: FloatArray,
    /** The saved voice they match, or null if they're not recognized. */
    val voice: VoiceProfile?,
    val score: Float,
    val role: Role,
    /** Talking in the last second or so (of live audio, which is ahead of what you hear). */
    val talkingNow: Boolean,
    /** Currently being covered. */
    val covered: Boolean,
    /** How many measurements the voiceprint is built from (more = more reliable). */
    val evidence: Int,
) {
    enum class Role { BLOCKED, ALLOWED, UNKNOWN }
}

interface BlockingListener {
    fun onSpeechChanged(speaking: Boolean) {}

    /** About 8 times a second: the mic's loudest recent level in dBFS, and the chance it's speech (0–1). */
    fun onLevel(dbfs: Float, speechProbability: Float) {}

    /** After each check: the closest blocked voice for the most recent speaker. */
    fun onAnalysis(match: VoiceMatch, blocking: Boolean) {}

    /**
     * A blocked person was found in audio that hasn't played yet. [leakedSeconds] is how much of their
     * speech had already played. [heard] is the voiceprint of what was heard, for "Not them".
     */
    fun onBlockStarted(voice: VoiceProfile, score: Float, leakedSeconds: Float, heard: FloatArray) {}
    fun onBlockEnded(voice: VoiceProfile) {}

    /** The phone couldn't analyze audio before it had to be played, so some went out unchecked. */
    fun onFellBehind() {}

    /** After each check (about once a second): everyone heard in the last little while, and who they are. */
    fun onSpeakers(speakers: List<LiveSpeaker>) {}
}

/**
 * Takes mic audio in 512-sample chunks and returns it [BlockingSettings.delaySeconds] later,
 * with blocked voices covered.
 *
 * Every second, on a background thread:
 *  1. pyannote segmentation looks at the last 10 s and marks when each of up to 3 people talks,
 *     including when they overlap.
 *  2. Each person still talking in audio that hasn't played yet gets a voiceprint from up to 3 s of
 *     their most recent speech where no one else was talking (WeSpeaker ResNet34).
 *  3. That voiceprint is added to a running track of the same voice, and the track is compared with
 *     the blocked and allowed voices. Longer tracks give steadier decisions.
 *  4. Unplayed audio where a blocked person talks is covered; where they overlap someone else it is
 *     turned down instead (or covered, per [OverlapStyle]).
 *  5. With a [separator], each stretch where a blocked person talks is split into two voices; the
 *     one that isn't them is played in place of the mix, so you still hear the other person.
 * Each stretch of audio is re-checked several times before it plays, and the latest, best-informed
 * decision wins.
 *
 * [process] must be called from a single audio thread. Other methods are thread-safe.
 */
class BlockingPipeline(
    private val encoder: VoiceEncoder,
    private val segmenter: SpeakerSegmenter,
    private val vad: VoiceActivityDetector,
    private val listener: BlockingListener? = null,
    /** Optional: separates Jerry from other voices instead of just turning overlaps down. */
    private val separator: SpeechSeparator? = null,
    private val analysisExecutor: Executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "earshield-analysis").apply { isDaemon = true; priority = Thread.NORM_PRIORITY + 1 }
    },
    private val random: Random = Random.Default,
) {
    @Volatile var settings = BlockingSettings()
    @Volatile var voices: List<VoiceProfile> = emptyList()

    private val window = SpeakerSegmenter.WINDOW_SAMPLES
    private val input = SampleRing(window + AudioFormat.samples(MAX_DELAY_SECONDS + 1f))
    /** Target gain for each sample in [input], written by the analysis thread. */
    private val gains = FloatArray(input.capacity) { 1f }
    /** Which track's voice caused each covered sample (−1 = none). */
    private val coveredBy = IntArray(input.capacity) { -1 }
    /** Where separation succeeded: the other voices with the blocked one removed, played instead of the mix. */
    private val replacement = FloatArray(input.capacity)
    private val replaced = BooleanArray(input.capacity)
    /** Covered samples where someone else was also heard: worth separating. */
    private val overlapped = BooleanArray(input.capacity)
    private var separatedUntil = 0L
    private val history = SampleRing(AudioFormat.samples(HISTORY_SECONDS))
    private var enrollment: SampleRing? = null
    /** Everything the mic hears while learning voices from a conversation (kept in memory only). */
    private var capture: SampleRing? = null

    private var received = 0L
    @Volatile private var playPos = 0L
    private var lastHopAt = 0L
    private val analysisBusy = AtomicBoolean(false)

    // Speech state from Silero VAD, used for enrollment, "use last 10 s" and the UI.
    private var inSpeech = false
    private var quietSamples = 0
    private val vadInput = FloatArray(AudioFormat.CHUNK)
    private var recentPeak = 0f
    private var levelChunks = 0
    private var levelMax = 0f
    private var probMax = 0f

    // Output
    private var gain = 1f
    private var repGain = 0f
    private var noise = 0f

    // Analysis thread only
    private val tracks = ArrayList<Track>()

    private class Track(val id: Int, var sum: FloatArray, var count: Int, var lastSeen: Long) {
        val centroid: FloatArray get() = Vectors.normalized(sum)
        var covering: VoiceProfile? = null
        /** The latest decision for this voice. */
        var blocked = false
    }
    private var nextTrackId = 0

    /** Processes one chunk of [AudioFormat.CHUNK] samples into [out] (same length). */
    fun process(chunk: FloatArray, out: FloatArray) {
        require(chunk.size == AudioFormat.CHUNK && out.size == chunk.size)
        val start = received
        for (i in chunk.indices) {
            val j = ((start + i) % gains.size).toInt()
            gains[j] = 1f
            coveredBy[j] = -1
            replaced[j] = false
            overlapped[j] = false
        }
        input.write(chunk)
        received += chunk.size
        synchronized(this) { capture?.let { if (it.size + chunk.size <= it.capacity) it.write(chunk) } }

        trackSpeech(chunk)

        if (received - lastHopAt >= AudioFormat.samples(HOP_SECONDS) && analysisBusy.compareAndSet(false, true)) {
            lastHopAt = received
            val end = received
            val snapshot = FloatArray(window) { input.at(end - window + it) }
            analysisExecutor.execute {
                try {
                    analyze(snapshot, end)
                } catch (e: Exception) {
                    e.printStackTrace()
                } finally {
                    analysisBusy.set(false)
                }
            }
        }
        render(out)
    }

    // --- Speech tracking (audio thread) ---------------------------------------------------------

    private fun trackSpeech(chunk: FloatArray) {
        val p = vad.speechProbability(boostForVad(chunk))
        reportLevel(chunk, p)
        synchronized(this) {
            if (!inSpeech && p >= SPEECH_ON) {
                inSpeech = true
                quietSamples = 0
                listener?.onSpeechChanged(true)
            }
            if (inSpeech) {
                history.write(chunk)
                enrollment?.let { if (it.size + chunk.size <= it.capacity) it.write(chunk) }
                quietSamples = if (p < SPEECH_OFF) quietSamples + chunk.size else 0
                if (quietSamples >= AudioFormat.samples(HANGOVER_SECONDS)) {
                    inSpeech = false
                    listener?.onSpeechChanged(false)
                }
            }
        }
    }

    /**
     * Phone mics often deliver speech far below full scale, and the speech detector misses quiet
     * speech. Boost the copy it sees so the loudest sound of the last few seconds sits near
     * −20 dBFS (at most +30 dB). The audio you hear is not changed.
     */
    private fun boostForVad(chunk: FloatArray): FloatArray {
        val rms = rms(chunk)
        recentPeak = max(rms, recentPeak * PEAK_DECAY)
        val g = (VAD_TARGET_RMS / recentPeak.coerceAtLeast(1e-6f)).coerceIn(1f, VAD_MAX_GAIN)
        for (i in chunk.indices) vadInput[i] = (chunk[i] * g).coerceIn(-1f, 1f)
        return vadInput
    }

    private fun reportLevel(chunk: FloatArray, p: Float) {
        levelMax = max(levelMax, rms(chunk))
        probMax = max(probMax, p)
        if (++levelChunks >= 4) {
            val db = if (levelMax > 0f) (20 * log10(levelMax.toDouble())).toFloat() else -120f
            listener?.onLevel(db, probMax)
            levelChunks = 0
            levelMax = 0f
            probMax = 0f
        }
    }

    private fun rms(x: FloatArray): Float {
        var s = 0f
        for (v in x) s += v * v
        return sqrt(s / x.size)
    }

    // --- Analysis (background thread) -----------------------------------------------------------

    /** What we know about one local speaker of the previous window, carried into the next one. */
    private class LocalSpeaker(
        val track: Track,
        /** Absolute sample where the solo speech used for the last voiceprint ended. */
        val embeddedUntil: Long,
        /** True if the last voiceprint had to use overlapped speech. */
        val fromOverlap: Boolean,
    )

    private var prevActivity: SpeakerSegmenter.Activity? = null
    private var prevWindowStart = 0L
    private var prevLocals = arrayOfNulls<LocalSpeaker>(SpeakerSegmenter.SPEAKERS)

    private fun analyze(audio: FloatArray, end: Long) {
        val s = settings
        val voices = voices
        val windowStart = end - audio.size
        // Leave a little room for audio that is being played right now.
        // (Early on, the window reaches back before listening started; those samples don't exist.)
        val decisionFrom = maxOf(playPos + AudioFormat.CHUNK * 2, windowStart, 0L)
        if (decisionFrom >= end) {
            listener?.onFellBehind()
            return
        }
        val act = segmenter.run(audio)
        val frames = act.frames
        fun sampleAt(f: Int): Long = windowStart + act.frameStart(f)
        val firstDecisionFrame = (0 until frames).firstOrNull { sampleAt(f = it + 1) > decisionFrom } ?: return
        val continuing = matchPreviousSpeakers(act, windowStart)

        // Each local speaker gets one or two "parts": normally one, but two if their voice changed
        // partway through (the segmentation model sometimes gives two people taking turns one label).
        class Part(val fromFrame: Int, val track: Track, val fromOverlap: Boolean) {
            var blocked = false
        }
        val parts = arrayOfNulls<List<Part>>(SpeakerSegmenter.SPEAKERS)
        val locals = arrayOfNulls<LocalSpeaker>(SpeakerSegmenter.SPEAKERS)

        for (k in 0 until SpeakerSegmenter.SPEAKERS) {
            val lastActive = (frames - 1 downTo 0).firstOrNull { act.active[k][it] } ?: continue
            val prev = continuing[k]
            var embeddedUntil = prev?.embeddedUntil ?: Long.MIN_VALUE
            val list: List<Part>
            if (prev == null || prev.fromOverlap) {
                val solo = speechOf(audio, act, k, soloOnly = true)
                list = when {
                    solo != null -> {
                        embeddedUntil = windowStart + solo.end
                        listOf(Part(0, assignTrack(encoder.embed(solo.audio), end, prev?.track), false))
                    }
                    prev != null -> listOf(Part(0, prev.track, true))
                    lastActive >= firstDecisionFrame -> {
                        // Someone new who has only talked over others so far: identify them from the
                        // overlapped speech. Less reliable, so it can only ever turn overlaps down.
                        val mixed = speechOf(audio, act, k, soloOnly = false) ?: continue
                        val t = Track(nextTrackId++, encoder.embed(mixed.audio).copyOf(), 1, end)
                        tracks += t
                        listOf(Part(0, t, true))
                    }
                    else -> continue
                }
            } else {
                // Re-measure only from speech that is new since last time, and only once there is
                // enough of it (more once the decision is clear). This keeps the phone's workload down.
                val needed = if (isConfident(prev.track, voices, s)) RECHECK_CONFIDENT_SECONDS else RECHECK_SECONDS
                val fromOffset = (prev.embeddedUntil - windowStart).coerceIn(0L, audio.size.toLong()).toInt()
                val fresh = speechOf(audio, act, k, soloOnly = true, fromOffset = fromOffset)
                list = if (fresh != null && fresh.audio.size >= AudioFormat.samples(needed)) {
                    val e = encoder.embed(fresh.audio)
                    val sim = Vectors.dot(e, prev.track.centroid)
                    val full = fresh.audio.size >= AudioFormat.samples(EMBED_MAX_SECONDS)
                    when {
                        sim >= TRACK_JOIN || (full && sim >= SPLIT_BELOW) -> {
                            embeddedUntil = windowStart + fresh.end
                            addToTrack(prev.track, e, end)
                            listOf(Part(0, prev.track, false))
                        }
                        sim < SPLIT_BELOW -> {
                            // Clearly a different voice under the same label: split where the new speech starts.
                            embeddedUntil = windowStart + fresh.end
                            val newTrack = assignTrack(e, end, null)
                            listOf(Part(0, prev.track, false), Part(fresh.startFrame, newTrack, false))
                        }
                        // Unclear: keep the current decision and wait for more speech before judging.
                        else -> listOf(Part(0, prev.track, false))
                    }
                } else {
                    listOf(Part(0, prev.track, false))
                }
            }
            parts[k] = list
            locals[k] = LocalSpeaker(list.last().track, embeddedUntil, list.last().fromOverlap)
        }
        prevActivity = act
        prevWindowStart = windowStart
        prevLocals = locals

        // Decide each part that talks in audio we can still change.
        var latestMatch: VoiceMatch? = null
        var latestEnd = -1
        val coveringNow = HashSet<Track>()
        for (k in 0 until SpeakerSegmenter.SPEAKERS) {
            val list = parts[k] ?: continue
            for ((i, part) in list.withIndex()) {
                val until = if (i + 1 < list.size) list[i + 1].fromFrame else frames
                val lastActive = (until - 1 downTo max(part.fromFrame, firstDecisionFrame)).firstOrNull { act.active[k][it] }
                    ?: continue
                val centroid = part.track.centroid
                val match = VoiceMatcher.best(centroid, voices)
                val voice = match.voice
                val threshold = if (part.fromOverlap) s.threshold - OVERLAP_RELAX else s.threshold
                val blockedVoice = voice != null && match.score >= threshold &&
                    match.score - match.allowedScore >= ALLOWED_MARGIN
                // "Block strangers": anyone who doesn't match an allowed voice is covered too.
                val stranger = s.blockStrangers && !blockedVoice && match.allowedScore < threshold
                part.blocked = blockedVoice || stranger
                val coverAs = if (blockedVoice) voice else if (stranger) STRANGER else null
                part.track.blocked = part.blocked
                if (DEBUG) println("  t=%.1f k=%d part=%d/%d overlapOnly=%s n=%d score=%.2f allowed=%.2f blocked=%s".format(
                    end / 16000f, k, i + 1, list.size, part.fromOverlap, part.track.count, match.score, match.allowedScore, part.blocked))
                if (lastActive > latestEnd) {
                    latestEnd = lastActive
                    latestMatch = match
                }
                if (part.blocked && coverAs != null) {
                    coveringNow += part.track
                    if (part.track.covering?.id != coverAs.id) {
                        part.track.covering = coverAs
                        var runStart = lastActive
                        while (runStart > part.fromFrame && act.active[k][runStart - 1]) runStart--
                        val leaked = max(0L, playPos - sampleAt(runStart))
                        val score = if (coverAs === STRANGER) match.allowedScore else match.score
                        listener?.onBlockStarted(coverAs, score, AudioFormat.seconds(leaked), centroid)
                    }
                }
            }
        }
        for (t in tracks) {
            val v = t.covering
            if (v != null && t !in coveringNow) {
                t.covering = null
                listener?.onBlockEnded(v)
            }
        }
        latestMatch?.let { listener?.onAnalysis(it, coveringNow.isNotEmpty()) }

        // Live speaker list: everyone heard recently, matched against saved voices.
        if (listener != null) {
            val talking = HashSet<Track>()
            val recentFrame = frames - (TALKING_NOW_SECONDS * frames * AudioFormat.SAMPLE_RATE / audio.size).toInt()
            for (k in 0 until SpeakerSegmenter.SPEAKERS) {
                val list = parts[k] ?: continue
                for ((i, part) in list.withIndex()) {
                    val until = if (i + 1 < list.size) list[i + 1].fromFrame else frames
                    if ((part.fromFrame until until).any { act.active[k][it] }) part.track.lastSeen = end
                    if ((max(part.fromFrame, recentFrame) until until).any { act.active[k][it] }) talking += part.track
                }
            }
            val live = tracks.filter { end - it.lastSeen <= AudioFormat.samples(LIVE_LIST_SECONDS) && it.count > 0 }
                .map { t ->
                    val c = t.centroid
                    val m = VoiceMatcher.best(c, voices)
                    val blockedVoice = m.voice != null && m.score >= s.threshold && m.score - m.allowedScore >= ALLOWED_MARGIN
                    val allowedVoice = !blockedVoice && m.allowedVoice != null && m.allowedScore >= s.threshold
                    LiveSpeaker(
                        trackId = t.id,
                        embedding = c,
                        voice = if (blockedVoice) m.voice else if (allowedVoice) m.allowedVoice else null,
                        score = if (blockedVoice) m.score else if (allowedVoice) m.allowedScore else max(m.score, m.allowedScore),
                        role = if (blockedVoice) LiveSpeaker.Role.BLOCKED else if (allowedVoice) LiveSpeaker.Role.ALLOWED else LiveSpeaker.Role.UNKNOWN,
                        talkingNow = t in talking,
                        covered = t.covering != null,
                        evidence = t.count,
                    )
                }
            listener.onSpeakers(live)
        }

        fun partAt(k: Int, f: Int): Part? = parts[k]?.lastOrNull { it.fromFrame <= f }

        // Where each blocked person talks, with short gaps filled in: the segmentation model often
        // loses the quieter voice for a moment during overlaps, and brief pauses shouldn't leak.
        val framesPerSecond = frames * AudioFormat.SAMPLE_RATE / audio.size.toFloat()
        val gapFill = (GAP_FILL_SECONDS * framesPerSecond).toInt()
        val blockedAct = Array(SpeakerSegmenter.SPEAKERS) { k ->
            val a = BooleanArray(frames) { f -> act.active[k][f] && partAt(k, f)?.blocked == true }
            var lastOn = -1
            for (f in 0 until frames) {
                if (a[f]) {
                    if (lastOn >= 0 && f - lastOn - 1 in 1..gapFill) for (g in lastOn + 1 until f) a[g] = true
                    lastOn = f
                }
            }
            a
        }

        // Write target gains for every unplayed sample in the window.
        val overlapGain = if (s.overlap == OverlapStyle.COVER) 0f else OVERLAP_GAIN
        val pad = (PAD_SECONDS * framesPerSecond).toInt()
        for (f in firstDecisionFrame until frames) {
            var blockedActive = false
            var blockedSoloCapable = false
            var otherActive = false
            var hitTrack = -1
            for (k in 0 until SpeakerSegmenter.SPEAKERS) {
                // Pad blocked speech a little: segment edges tend to clip the first and last sounds.
                val lo = max(0, f - pad)
                val hi = min(frames - 1, f + pad)
                var hit: Part? = null
                for (g in lo..hi) if (blockedAct[k][g]) { hit = partAt(k, g); break }
                if (hit != null) {
                    blockedActive = true
                    hitTrack = hit.track.id
                    if (!hit.fromOverlap) blockedSoloCapable = true
                } else if (act.active[k][f]) {
                    otherActive = true
                }
            }
            val g = when {
                !blockedActive -> 1f
                otherActive || !blockedSoloCapable -> overlapGain
                else -> 0f
            }
            val from = max(sampleAt(f), decisionFrom)
            val to = if (f + 1 < frames) sampleAt(f + 1) else end
            for (i in from until to) {
                val j = (i % gains.size).toInt()
                if (g < 1f) {
                    gains[j] = min(g, if (coveredBy[j] >= 0) gains[j] else 1f)
                    coveredBy[j] = hitTrack
                    if (otherActive) overlapped[j] = true
                } else {
                    // An earlier check heard a blocked voice here. If that voice is still judged
                    // blocked, keep covering: the segmentation model can lose a voice mid-overlap.
                    // If new evidence says it wasn't them after all, uncover.
                    val by = coveredBy[j]
                    if (by < 0 || tracks.none { it.id == by && it.blocked }) {
                        gains[j] = 1f
                        coveredBy[j] = -1
                        replaced[j] = false
                    }
                }
            }
        }
        if (separator != null && s.blockingEnabled && s.overlap == OverlapStyle.SEPARATE) separate(audio, windowStart, max(decisionFrom, separatedUntil), end, voices, s)
        tracks.removeAll { end - it.lastSeen > AudioFormat.samples(TRACK_MEMORY_SECONDS) && it.covering == null }
    }

    /**
     * For each stretch in [from, end) where a blocked voice was heard, split the audio into two
     * voices, identify the blocked one, and play the other in its place. If both streams sound like
     * the blocked person (they were talking alone), keep covering instead.
     */
    private fun separate(audio: FloatArray, windowStart: Long, from: Long, end: Long, voices: List<VoiceProfile>, s: BlockingSettings) {
        val sep = separator ?: return
        val blockedVoices = voices.filter { !it.allowed }
        val allowedVoices = voices.filter { it.allowed }
        if (if (s.blockStrangers) allowedVoices.isEmpty() else blockedVoices.isEmpty()) return
        val merge = AudioFormat.samples(SEP_MERGE_SECONDS)
        val runs = ArrayList<LongArray>()
        var i = max(from, 0L)
        while (i < end) {
            if (coveredBy[(i % gains.size).toInt()] >= 0) {
                var j = i
                while (j < end && coveredBy[(j % gains.size).toInt()] >= 0) j++
                val last = runs.lastOrNull()
                if (last != null && i - last[1] <= merge) last[1] = j else runs += longArrayOf(i, j)
                i = j
            } else {
                i++
            }
        }
        for ((a, b) in runs.map { it[0] to it[1] }) {
            if (b - a < AudioFormat.samples(SEP_MIN_SECONDS)) continue
            // Separating is costly; only do it where someone else was heard too.
            var anyOverlap = false
            for (n in a until b) if (overlapped[(n % gains.size).toInt()]) { anyOverlap = true; break }
            if (!anyOverlap) continue
            val ctxStart = maxOf(a - AudioFormat.samples(SEP_CONTEXT_SECONDS), windowStart, 0L)
            val chunk = audio.copyOfRange((ctxStart - windowStart).toInt(), (b - windowStart).toInt())
            val streams = sep.separate(chunk)
            scaleToMix(chunk, streams)
            val embeds = Array(2) { k -> encoder.embed(streams[k]) }
            val near = s.threshold - SEP_SAME_MARGIN
            val other: FloatArray
            val swap: Boolean
            val alone: Boolean
            val score: FloatArray
            val jerry: Int
            if (s.blockStrangers) {
                // Keep the stream that is an allowed voice; cover if neither is.
                score = FloatArray(2) { k -> allowedVoices.maxOf { v -> Vectors.dot(embeds[k], v.embedding) } }
                val keep = if (score[0] >= score[1]) 0 else 1
                jerry = 1 - keep
                other = streams[keep]
                val keepLevel = rms(other) / rms(chunk).coerceAtLeast(1e-9f)
                val keepOk = score[keep] >= near && keepLevel >= SEP_MIN_LEVEL
                if (keepOk && score[jerry] >= near) continue // both allowed: nothing to remove
                swap = keepOk
                alone = !keepOk
            } else {
                score = FloatArray(2) { k -> blockedVoices.maxOf { v -> Vectors.dot(embeds[k], v.embedding) } }
                jerry = if (score[0] >= score[1]) 0 else 1
                other = streams[1 - jerry]
                val level = rms(other) / rms(chunk).coerceAtLeast(1e-9f)
                // Only swap in the other stream when the split is clear: one stream is recognizably the
                // blocked person and the other isn't. If both sound like them, they were talking alone:
                // cover. If neither is recognizable (too little audio), keep the cover/turn-down decision.
                val recognized = score[jerry] >= near
                alone = recognized && (score[1 - jerry] >= near || level < SEP_MIN_LEVEL)
                swap = recognized && !alone
            }
            val otherLevel = rms(other) / rms(chunk).coerceAtLeast(1e-9f)
            if (DEBUG) println("  separate %.1f–%.1f s: scores %.2f/%.2f, other level %.2f -> %s".format(
                a / 16000f, b / 16000f, score[jerry], score[1 - jerry], otherLevel,
                if (swap) "replace" else if (alone) "cover" else "unclear, keep"))
            if (!swap && !alone) continue
            val guard = playPos + AudioFormat.CHUNK * 2
            for (n in max(a, guard) until b) {
                val j = (n % gains.size).toInt()
                if (alone) {
                    gains[j] = 0f
                } else {
                    replacement[j] = other[(n - ctxStart).toInt()]
                    replaced[j] = true
                }
            }
        }
        separatedUntil = max(separatedUntil, runs.lastOrNull()?.get(1) ?: from)
    }

    /** Rescales the two streams so that together they best reproduce the mix (least squares). */
    private fun scaleToMix(mix: FloatArray, s: Array<FloatArray>) {
        var a11 = 0.0; var a12 = 0.0; var a22 = 0.0; var b1 = 0.0; var b2 = 0.0
        for (n in mix.indices) {
            val x = s[0][n].toDouble(); val y = s[1][n].toDouble(); val m = mix[n].toDouble()
            a11 += x * x; a12 += x * y; a22 += y * y; b1 += x * m; b2 += y * m
        }
        val det = a11 * a22 - a12 * a12
        if (abs(det) < 1e-12) return
        val c1 = ((b1 * a22 - b2 * a12) / det).toFloat()
        val c2 = ((a11 * b2 - a12 * b1) / det).toFloat()
        for (n in mix.indices) { s[0][n] *= c1; s[1][n] *= c2 }
    }

    /** A track is "confident" when its decision is well clear of the threshold either way. */
    private fun isConfident(track: Track, voices: List<VoiceProfile>, s: BlockingSettings): Boolean {
        if (track.count < 3) return false
        val m = VoiceMatcher.best(track.centroid, voices)
        return m.voice == null || abs(m.score - s.threshold) >= CONFIDENT_MARGIN
    }

    /**
     * Local speaker labels are only consistent within one window. Match each speaker in this window
     * to the one in the previous window whose activity lines up best over the stretch both cover.
     */
    private fun matchPreviousSpeakers(act: SpeakerSegmenter.Activity, windowStart: Long): Array<LocalSpeaker?> {
        val result = arrayOfNulls<LocalSpeaker>(SpeakerSegmenter.SPEAKERS)
        val prev = prevActivity ?: return result
        val shift = windowStart - prevWindowStart
        if (shift <= 0 || shift >= prev.windowSamples) return result
        val used = BooleanArray(SpeakerSegmenter.SPEAKERS)
        for (k in 0 until SpeakerSegmenter.SPEAKERS) {
            var bestJ = -1
            var bestCount = 0
            var activeInOverlap = 0
            for (f in 0 until act.frames) {
                val prevSample = act.frameStart(f) + shift
                if (prevSample >= prev.windowSamples) break
                if (act.active[k][f]) activeInOverlap++
            }
            for (j in 0 until SpeakerSegmenter.SPEAKERS) {
                if (used[j] || prevLocals[j] == null) continue
                var both = 0
                for (f in 0 until act.frames) {
                    val pf = ((act.frameStart(f) + shift) * prev.frames / prev.windowSamples).toInt()
                    if (pf >= prev.frames) break
                    if (act.active[k][f] && prev.active[j][pf]) both++
                }
                if (both > bestCount) {
                    bestCount = both
                    bestJ = j
                }
            }
            if (bestJ >= 0 && bestCount >= MIN_CONTINUITY_FRAMES && bestCount * 2 >= activeInOverlap) {
                used[bestJ] = true
                result[k] = prevLocals[bestJ]
            }
        }
        return result
    }

    private class Speech(val audio: FloatArray, val startFrame: Int, val end: Int)

    /**
     * Up to [EMBED_MAX_SECONDS] of speaker [k]'s most recent speech after window offset [fromOffset]
     * (only where nobody else talks if [soloOnly]), or null if there is less than [EMBED_MIN_SECONDS].
     */
    private fun speechOf(
        audio: FloatArray,
        act: SpeakerSegmenter.Activity,
        k: Int,
        soloOnly: Boolean,
        fromOffset: Int = 0,
    ): Speech? {
        fun ok(f: Int) = act.active[k][f] && (!soloOnly || act.speakersAt(f) == 1) && act.frameStart(f) >= fromOffset
        val maxSamples = AudioFormat.samples(EMBED_MAX_SECONDS)
        val pieces = ArrayList<IntRange>()
        var total = 0
        var lastEnd = -1
        var firstFrame = -1
        var f = act.frames - 1
        while (f >= 0 && total < maxSamples) {
            if (ok(f)) {
                var start = f
                while (start > 0 && ok(start - 1)) start--
                val from = act.frameStart(start)
                val to = if (f + 1 < act.frames) act.frameStart(f + 1) else audio.size
                if (lastEnd < 0) lastEnd = to
                val take = min(to - from, maxSamples - total)
                pieces += (to - take) until to
                total += take
                firstFrame = start
                f = start - 1
            } else {
                f--
            }
        }
        if (total < AudioFormat.samples(EMBED_MIN_SECONDS)) return null
        val out = FloatArray(total)
        var o = 0
        for (r in pieces.asReversed()) {
            System.arraycopy(audio, r.first, out, o, r.last - r.first + 1)
            o += r.last - r.first + 1
        }
        return Speech(out, firstFrame, lastEnd)
    }

    /** Adds a voiceprint to [preferred] (the same person as last window) or the best matching track, or starts one. */
    private fun assignTrack(embedding: FloatArray, now: Long, preferred: Track?): Track {
        var best: Track? = null
        if (preferred != null && Vectors.dot(embedding, preferred.centroid) >= TRACK_JOIN) {
            best = preferred
        } else {
            // Rejoining an older voice needs a strong match; short clips in a noisy room can
            // look alike, and wrongly merging two people is worse than starting fresh.
            var bestSim = TRACK_REJOIN
            for (t in tracks) {
                val sim = Vectors.dot(embedding, t.centroid)
                if (sim >= bestSim) {
                    bestSim = sim
                    best = t
                }
            }
        }
        val track = best ?: Track(nextTrackId++, FloatArray(embedding.size), 0, now).also { tracks += it }
        addToTrack(track, embedding, now)
        return track
    }

    /** Running sum that slowly forgets, so a track follows its voice without drifting to others. */
    private fun addToTrack(track: Track, embedding: FloatArray, now: Long) {
        val decay = if (track.count >= TRACK_MAX) (TRACK_MAX - 1f) / TRACK_MAX else 1f
        for (i in embedding.indices) track.sum[i] = track.sum[i] * decay + embedding[i]
        track.count = min(track.count + 1, TRACK_MAX)
        track.lastSeen = now
    }

    // --- Output (audio thread) ------------------------------------------------------------------

    private fun render(out: FloatArray) {
        val s = settings
        val delay = AudioFormat.samples(s.delaySeconds.coerceIn(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
        var index = received - out.size - delay
        playPos = index + out.size
        for (i in out.indices) {
            val x = input.at(index)
            val j = if (index >= 0) (index % gains.size).toInt() else -1
            val useRep = s.blockingEnabled && j >= 0 && replaced[j]
            val target = if (useRep) 0f else if (s.blockingEnabled && j >= 0) gains[j] else 1f
            gain = when {
                gain < target -> min(target, gain + 1f / FADE_SAMPLES)
                gain > target -> max(target, gain - 1f / FADE_SAMPLES)
                else -> gain
            }
            val repTarget = if (useRep) 1f else 0f
            repGain = when {
                repGain < repTarget -> min(repTarget, repGain + 1f / FADE_SAMPLES)
                repGain > repTarget -> max(repTarget, repGain - 1f / FADE_SAMPLES)
                else -> repGain
            }
            val rep = if (j >= 0 && repGain > 0f) replacement[j] * repGain else 0f
            val covered = (1 - gain - repGain).coerceAtLeast(0f) // the part neither played nor replaced
            out[i] = rep + when (s.cover) {
                CoverStyle.SILENCE -> x * gain
                CoverStyle.QUIETER -> x * (gain + QUIET_GAIN * covered)
                CoverStyle.SOFT_NOISE -> x * gain + covered * nextNoise()
            }
            index++
        }
    }

    /** Low-passed white noise: a soft "shhh" rather than a harsh hiss. */
    private fun nextNoise(): Float {
        noise = 0.95f * noise + 0.05f * (random.nextFloat() * 2 - 1)
        return noise * NOISE_SCALE
    }

    // --- Enrollment -----------------------------------------------------------------------------

    /** Starts recording everything the mic hears, for learning voices from a conversation. */
    @Synchronized fun startCapture(maxSeconds: Float) {
        capture = SampleRing(AudioFormat.samples(maxSeconds))
    }

    /** Seconds recorded so far, or null if not recording. */
    @Synchronized fun captureSeconds(): Float? = capture?.let { AudioFormat.seconds(it.size.toLong()) }

    /** Stops recording and returns the audio (null if not recording). */
    @Synchronized fun finishCapture(): FloatArray? {
        val c = capture ?: return null
        capture = null
        return c.latest(c.size)
    }

    @Synchronized fun cancelCapture() {
        capture = null
    }

    /** Starts collecting speech for a new voiceprint. */
    @Synchronized fun startEnrollment() {
        enrollment = SampleRing(AudioFormat.samples(Enrollment.MAX_SECONDS))
    }

    /** Seconds of speech collected so far, or null if not enrolling. */
    @Synchronized fun enrollmentSeconds(): Float? = enrollment?.let { AudioFormat.seconds(it.size.toLong()) }

    /** Stops collecting and returns the speech collected (null if not enrolling). */
    @Synchronized fun finishEnrollment(): FloatArray? {
        val e = enrollment ?: return null
        enrollment = null
        return e.latest(e.size)
    }

    @Synchronized fun cancelEnrollment() {
        enrollment = null
    }

    /** The most recent [seconds] of speech heard, for "block whoever just spoke". */
    @Synchronized fun recentSpeech(seconds: Float): FloatArray = history.latest(AudioFormat.samples(seconds))

    companion object {
        /** Stands in for "someone who isn't on the allow list" in block events. */
        val STRANGER = VoiceProfile("stranger", "Someone not on your allow list", FloatArray(0))
        const val MIN_DELAY_SECONDS = 1f
        const val TALKING_NOW_SECONDS = 1.5f
        const val LIVE_LIST_SECONDS = 30f
        const val MAX_DELAY_SECONDS = 5f
        private val DEBUG = System.getProperty("earshield.debug") == "1"
        const val HOP_SECONDS = 1.0f
        const val RECHECK_SECONDS = 1.5f // new solo speech needed before re-measuring an undecided voice
        const val SPLIT_BELOW = 0.25f // new speech this unlike the current voice means someone else took over
        const val RECHECK_CONFIDENT_SECONDS = 3.0f // ... or a voice that is clearly blocked or clearly not
        const val CONFIDENT_MARGIN = 0.12f
        const val MIN_CONTINUITY_FRAMES = 20
        const val OVERLAP_RELAX = 0.10f // lower threshold when only overlapped speech is available
        const val EMBED_MIN_SECONDS = 1.0f
        const val EMBED_MAX_SECONDS = 3.0f
        const val PAD_SECONDS = 0.15f
        const val GAP_FILL_SECONDS = 0.6f
        const val SEP_CONTEXT_SECONDS = 1.0f // extra audio before each stretch, so separation can settle
        const val SEP_MIN_SECONDS = 0.3f
        const val SEP_MERGE_SECONDS = 0.5f
        const val SEP_SAME_MARGIN = 0.10f
        const val SEP_MIN_LEVEL = 0.05f // other stream below about −26 dB of the mix means nobody else is talking
        const val OVERLAP_GAIN = 0.25f // about −12 dB
        const val ALLOWED_MARGIN = 0.05f
        const val TRACK_JOIN = 0.40f // a new voiceprint joins a track if at least this similar (same person: usually > 0.36; others: < 0.33)
        const val TRACK_REJOIN = 0.55f // a voice reappearing after a gap must match an old track this well
        const val TRACK_MAX = 12
        const val TRACK_MEMORY_SECONDS = 120f
        const val SPEECH_ON = 0.5f
        const val SPEECH_OFF = 0.35f
        const val HANGOVER_SECONDS = 0.35f
        const val HISTORY_SECONDS = 20f
        const val FADE_SAMPLES = 160 // 10 ms
        const val QUIET_GAIN = 0.08f
        const val NOISE_SCALE = 0.15f
        const val VAD_TARGET_RMS = 0.1f
        const val VAD_MAX_GAIN = 31.6f
        const val PEAK_DECAY = 0.995f
    }
}
