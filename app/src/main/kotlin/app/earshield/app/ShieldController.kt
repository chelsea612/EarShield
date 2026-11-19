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
        BlockingSettings(delaySeconds, threshold, cover, overlap, blockingEnabled = !paused)

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
        return BlockingPipeline(
            enc,
            SpeakerSegmenter(bytes("speaker_segmentation.onnx")),
            SileroVad(bytes("silero_vad.onnx")),
            listener = this,
            separator = SpeechSeparator(bytes("speech_separator.onnx"), threads = 2),
        )
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
        voiceStore.add(name.trim().ifEmpty { if (allowed) "You" else "Blocked voice" }, result.embedding, allowed)
    }

    /** "That wasn't them": remember what was heard as a voice that is never blocked. */
    fun markNotThem(eventId: Long) {
        val event = _status.value.recentBlocks.firstOrNull { it.id == eventId } ?: return
        if (event.corrected) return
        val time = java.text.DateFormat.getTimeInstance(java.text.DateFormat.SHORT).format(java.util.Date(event.timeMillis))
        voiceStore.add("Not ${event.name} ($time)", event.heard, allowed = true)
        _status.update { st -> st.copy(recentBlocks = st.recentBlocks.map { if (it.id == eventId) it.copy(corrected = true) else it }) }
    }

    fun removeVoice(id: String) = voiceStore.remove(id)
}
