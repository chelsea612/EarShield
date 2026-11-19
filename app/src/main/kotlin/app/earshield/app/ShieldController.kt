package app.earshield.app

import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import androidx.core.content.ContextCompat
import app.earshield.core.BlockingListener
import app.earshield.core.BlockingPipeline
import app.earshield.core.BlockingSettings
import app.earshield.core.Enrollment
import app.earshield.core.SpeakerSegmenter
import app.earshield.core.Vectors
import app.earshield.core.VoiceDiscovery
import app.earshield.core.SpeechSeparator
import app.earshield.core.WespeakerEncoder
import app.earshield.core.SileroVad
import app.earshield.core.VoiceMatch
import app.earshield.core.VoiceProfile
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class BlockEvent(
    val id: Long,
    val name: String,
    val timeMillis: Long,
    val leakedSeconds: Float,
    val score: Float,
    /** Voiceprint of what was heard, saved as an allowed voice if you tap "Not them". */
    val heard: FloatArray,
    val corrected: Boolean = false,
) {
    fun copy(corrected: Boolean) = BlockEvent(id, name, timeMillis, leakedSeconds, score, heard, corrected)
}

enum class VoiceRole { BLOCK, ALLOW, ME, SKIP }

/** A person found in a recorded conversation, waiting for you to label them. */
class DiscoveredVoice(
    val number: Int,
    val found: VoiceDiscovery.FoundVoice,
    /** An already-saved voice this sounds like, if any. */
    val knownAs: VoiceProfile?,
    val knownScore: Float,
    /** Another found voice (by number) that this is probably the same person as. */
    val maybeSameAs: Int?,
)

sealed interface SaveOutcome {
    data object Saved : SaveOutcome
    data class Failed(val message: String) : SaveOutcome
    /** The recording seems to contain more than one voice; [saveAnyway] can still keep it. */
    data object Mixed : SaveOutcome
}

data class LiveStatus(
    val running: Boolean = false,
    val starting: Boolean = false,
    val paused: Boolean = false,
    val speaking: Boolean = false,
    val closestName: String? = null,
    val closestScore: Float = 0f,
    val blockingName: String? = null,
    val enrollingSeconds: Float? = null,
    /** Seconds of conversation recorded so far while learning voices, or null. */
    val captureSeconds: Float? = null,
    /** 0–1 while finding voices in a recording, or null. */
    val discoveryProgress: Float? = null,
    /** Mic level in dBFS and speech probability, updated about 8 times a second. */
    val micDb: Float = -120f,
    val speechProbability: Float = 0f,
    /** The mic is delivering pure silence: access blocked, or another app is using it. */
    val micSilent: Boolean = false,
    val recentBlocks: List<BlockEvent> = emptyList(),
    val error: String? = null,
    /** When the phone last couldn't analyze audio in time (some audio played unchecked). */
    val fellBehindAtMillis: Long = 0L,
    /** Output silenced because no earbuds/headphones are connected. */
    val outputMuted: Boolean = false,
    /** Diagnostics: where sound is going, and playback gaps since starting. */
    val outputRoute: String = "",
    val playbackGaps: Int = 0,
)

/** App-wide owner of the models, the pipeline and the audio engine. */
class ShieldController(private val context: Context) : BlockingListener {
    val settingsStore = SettingsStore(context)
    val voiceStore = VoiceStore(context)

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val _status = MutableStateFlow(LiveStatus())
    private val paused = MutableStateFlow(false)
    val status: StateFlow<LiveStatus> = _status.asStateFlow()

    private var encoder: WespeakerEncoder? = null
    private var segmenter: SpeakerSegmenter? = null
    private var separator: SpeechSeparator? = null
    private var captureJob: Job? = null
    private var pipeline: BlockingPipeline? = null
    private var engine: AudioEngine? = null
    private var engineMode: InputMode? = null
    private var enrollmentJob: Job? = null
    @Volatile private var nextEventId = 0L
    private var pending: Enrollment.Result? = null

    init {
        scope.launch {
            combine(settingsStore.settings, paused) { s, p -> s to p }.collect { (s, paused) ->
                pipeline?.settings = s.toBlocking(paused)
                if (engine != null && engineMode != s.inputMode) restartEngine()
            }
        }
        scope.launch { voiceStore.voices.collect { pipeline?.voices = it } }
    }

    private fun AppSettings.toBlocking(paused: Boolean) =
        BlockingSettings(delaySeconds, threshold, cover, overlap, blockingEnabled = !paused, blockStrangers = blockStrangers)

    // --- Start / stop -----------------------------------------------------------------------

    /** Starts listening. The caller must already hold RECORD_AUDIO. */
    fun start() {
        _status.update { it.copy(starting = true, error = null) }
        ContextCompat.startForegroundService(context, Intent(context, ShieldService::class.java))
    }

    fun stop() {
        context.stopService(Intent(context, ShieldService::class.java))
    }

    fun setPaused(value: Boolean) {
        paused.value = value
        _status.update { it.copy(paused = value) }
    }

    /** Called by [ShieldService] once it is in the foreground. */
    internal fun startAudio() {
        scope.launch(Dispatchers.IO) {
            try {
                val p = pipeline ?: createPipeline().also { pipeline = it }
                p.settings = settingsStore.settings.value.toBlocking(paused.value)
                p.voices = voiceStore.voices.value
                synchronized(this@ShieldController) { startEngine(p) }
                _status.update { it.copy(running = true, starting = false) }
            } catch (e: Exception) {
                fail("Couldn't start: ${e.message}")
            }
        }
    }

    /** Called by [ShieldService] when it is destroyed. */
    internal fun stopAudio() {
        synchronized(this) {
            engine?.stop()
            engine = null
            engineMode = null
        }
        cancelEnrollment()
        cancelCapture()
        _status.update {
            it.copy(running = false, starting = false, speaking = false, blockingName = null,
                micDb = -120f, speechProbability = 0f, micSilent = false, outputMuted = false)
        }
    }

    private fun createPipeline(): BlockingPipeline {
        val assets = context.assets
        fun bytes(name: String) = assets.open(name).use { it.readBytes() }
        val enc = WespeakerEncoder(bytes("speaker_encoder.onnx"))
        encoder = enc
        val seg = SpeakerSegmenter(bytes("speaker_segmentation.onnx")).also { segmenter = it }
        val sep = SpeechSeparator(bytes("speech_separator.onnx"), threads = 2).also { separator = it }
        return BlockingPipeline(enc, seg, SileroVad(bytes("silero_vad.onnx")), listener = this, separator = sep)
    }

    private fun startEngine(p: BlockingPipeline) {
        val mode = settingsStore.settings.value.inputMode
        engine?.stop()
        engine = AudioEngine(context, p, mode, onError = { msg -> fail(msg) },
            onSilentMic = { silent -> _status.update { it.copy(micSilent = silent) } },
            onPrivateOutput = { ok -> _status.update { it.copy(outputMuted = !ok) } },
            onOutputInfo = { route, gaps -> _status.update { it.copy(outputRoute = route, playbackGaps = gaps) } }).also { it.start() }
        engineMode = mode
    }

    private fun restartEngine() {
        val p = pipeline ?: return
        scope.launch(Dispatchers.IO) { synchronized(this@ShieldController) { if (engine != null) startEngine(p) } }
    }

    private fun fail(message: String) {
        _status.update { it.copy(error = message, starting = false) }
        stop()
    }

    fun clearError() = _status.update { it.copy(error = null) }

    // --- Pipeline events (called on the audio thread) -------------------------------------------

    override fun onSpeechChanged(speaking: Boolean) {
        _status.update { if (speaking) it.copy(speaking = true) else it.copy(speaking = false, closestName = null, closestScore = 0f) }
    }

    override fun onLevel(dbfs: Float, speechProbability: Float) {
        _status.update { it.copy(micDb = dbfs, speechProbability = speechProbability) }
    }

    override fun onAnalysis(match: VoiceMatch, blocking: Boolean) {
        _status.update { it.copy(closestName = match.voice?.name, closestScore = match.score) }
    }

    override fun onBlockStarted(voice: VoiceProfile, score: Float, leakedSeconds: Float, heard: FloatArray) {
        val event = BlockEvent(nextEventId++, voice.name, System.currentTimeMillis(), leakedSeconds, score, heard)
        _status.update { it.copy(blockingName = voice.name, recentBlocks = (listOf(event) + it.recentBlocks).take(20)) }
        if (settingsStore.settings.value.buzzOnBlock && !paused.value) buzz()
    }

    override fun onFellBehind() {
        _status.update { it.copy(fellBehindAtMillis = System.currentTimeMillis()) }
    }

    override fun onBlockEnded(voice: VoiceProfile) {
        _status.update { it.copy(blockingName = null) }
    }

    private fun buzz() {
        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            context.getSystemService(VibratorManager::class.java).defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Vibrator::class.java)
        }
        vibrator?.vibrate(VibrationEffect.createOneShot(40, VibrationEffect.DEFAULT_AMPLITUDE))
    }

    // --- Enrollment -----------------------------------------------------------------------------

    /** Starts recording a new voice. Listening must be running. */
    fun startEnrollment() {
        val p = pipeline ?: return
        p.startEnrollment()
        enrollmentJob?.cancel()
        enrollmentJob = scope.launch {
            while (isActive) {
                _status.update { it.copy(enrollingSeconds = p.enrollmentSeconds()) }
                delay(200)
            }
        }
    }

    fun cancelEnrollment() {
        enrollmentJob?.cancel()
        enrollmentJob = null
        pipeline?.cancelEnrollment()
        _status.update { it.copy(enrollingSeconds = null) }
    }

    /** Finishes recording and saves the voice ([allowed] = never block it, e.g. your own voice). */
    suspend fun finishEnrollment(name: String, allowed: Boolean): SaveOutcome {
        enrollmentJob?.cancel()
        enrollmentJob = null
        val speech = pipeline?.finishEnrollment()
        _status.update { it.copy(enrollingSeconds = null) }
        return save(name, speech, allowed)
    }

    /** Learns whoever spoke most recently, using the last [seconds] of speech. */
    suspend fun enrollFromRecent(name: String, allowed: Boolean, seconds: Float = 10f): SaveOutcome =
        save(name, pipeline?.recentSpeech(seconds), allowed)

    private suspend fun save(name: String, speech: FloatArray?, allowed: Boolean): SaveOutcome =
        withContext(Dispatchers.Default) {
            val enc = encoder ?: return@withContext SaveOutcome.Failed("Start listening first.")
            val result = speech?.let { Enrollment.analyze(enc, it) }
                ?: return@withContext SaveOutcome.Failed(
                    "Not enough speech yet. It needs at least ${Enrollment.MIN_SECONDS.toInt()} seconds of talking.")
            if (result.looksMixed) {
                pending = result
                return@withContext SaveOutcome.Mixed
            }
            store(name, result, allowed)
            SaveOutcome.Saved
        }

    /** Keeps a recording that [SaveOutcome.Mixed] warned about. */
    fun saveAnyway(name: String, allowed: Boolean) {
        pending?.let { store(name, it, allowed) }
        pending = null
    }

    private fun store(name: String, result: Enrollment.Result, allowed: Boolean) {
        voiceStore.add(name.trim().ifEmpty { if (allowed) "Allowed voice" else "Blocked voice" }, result.embedding, allowed)
    }

    /** "That wasn't them": remember what was heard as a voice that is never blocked. */
    fun markNotThem(eventId: Long) {
        val event = _status.value.recentBlocks.firstOrNull { it.id == eventId } ?: return
        if (event.corrected) return
        val time = java.text.DateFormat.getTimeInstance(java.text.DateFormat.SHORT).format(java.util.Date(event.timeMillis))
        val name = if (event.name == BlockingPipeline.STRANGER.name) "Allowed voice ($time)" else "Not ${event.name} ($time)"
        voiceStore.add(name, event.heard, allowed = true)
        _status.update { st -> st.copy(recentBlocks = st.recentBlocks.map { if (it.id == eventId) it.copy(corrected = true) else it }) }
    }

    fun removeVoice(id: String) = voiceStore.remove(id)
    fun updateVoice(id: String, name: String, allowed: Boolean) = voiceStore.update(id, name, allowed)
    fun mergeVoices(keepId: String, otherId: String, name: String, allowed: Boolean) =
        voiceStore.merge(keepId, otherId, name, allowed)

    // --- Learning voices from a conversation -------------------------------------------------------

    /** Starts recording the conversation (up to 90 s, in memory only). Listening must be running. */
    fun startCapture() {
        val p = pipeline ?: return
        p.startCapture(VoiceDiscovery.MAX_RECORDING_SECONDS)
        captureJob?.cancel()
        captureJob = scope.launch {
            while (isActive) {
                _status.update { it.copy(captureSeconds = p.captureSeconds()) }
                delay(250)
            }
        }
    }

    fun cancelCapture() {
        captureJob?.cancel()
        captureJob = null
        pipeline?.cancelCapture()
        _status.update { it.copy(captureSeconds = null, discoveryProgress = null) }
    }

    /** Stops recording and finds the people in it. The recording is dropped once this returns. */
    suspend fun findVoices(): List<DiscoveredVoice> {
        captureJob?.cancel()
        captureJob = null
        val audio = pipeline?.finishCapture()
        _status.update { it.copy(captureSeconds = null, discoveryProgress = 0f) }
        val enc = encoder
        val seg = segmenter
        if (audio == null || enc == null || seg == null) {
            _status.update { it.copy(discoveryProgress = null) }
            return emptyList()
        }
        return withContext(Dispatchers.Default) {
            val found = VoiceDiscovery(enc, seg, separator).discover(audio) { p ->
                _status.update { it.copy(discoveryProgress = p) }
            }
            val known = voiceStore.voices.value.filterNot { it.name.startsWith("Not ") || it.name.startsWith("Allowed voice (") }
            val threshold = settingsStore.settings.value.threshold
            val result = found.mapIndexed { i, v ->
                val best = known.maxByOrNull { Vectors.dot(v.embedding, it.embedding) }
                val bestScore = best?.let { Vectors.dot(v.embedding, it.embedding) } ?: 0f
                val sameAs = found.indices.filter { it < i }.firstOrNull { j ->
                    Vectors.dot(v.embedding, found[j].embedding) >= POSSIBLY_SAME
                }
                DiscoveredVoice(
                    number = i + 1,
                    found = v,
                    knownAs = if (best != null && bestScore >= threshold) best else null,
                    knownScore = bestScore,
                    maybeSameAs = sameAs?.plus(1),
                )
            }
            _status.update { it.copy(discoveryProgress = null) }
            result
        }
    }

    /** Plays a found voice's sample through the earbuds (never the phone speaker). */
    fun playClip(clip: FloatArray) {
        var sq = 0.0
        for (x in clip) sq += x * x
        val rms = kotlin.math.sqrt(sq / clip.size.coerceAtLeast(1)).toFloat().coerceAtLeast(1e-6f)
        val gain = (0.08f / rms).coerceAtMost(30f)
        engine?.playPreview(FloatArray(clip.size) { (clip[it] * gain).coerceIn(-1f, 1f) })
    }

    /** Saves the voices you labeled. */
    fun saveDiscovered(choices: List<Triple<DiscoveredVoice, String, VoiceRole>>) {
        for ((v, name, role) in choices) {
            when (role) {
                VoiceRole.SKIP -> Unit
                VoiceRole.BLOCK -> voiceStore.add(name.trim().ifEmpty { "Blocked voice ${v.number}" }, v.found.embedding, allowed = false)
                VoiceRole.ALLOW -> voiceStore.add(name.trim().ifEmpty { "Allowed voice ${v.number}" }, v.found.embedding, allowed = true)
                VoiceRole.ME -> voiceStore.add("You", v.found.embedding, allowed = true)
            }
        }
    }

    private companion object {
        /** Two found voices this alike are probably one person who got split up. */
        const val POSSIBLY_SAME = 0.45f
    }
}
