package app.earshield.app

import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Build
import android.os.Process
import android.os.SystemClock
import android.util.Log
import androidx.core.content.ContextCompat
import app.earshield.core.BlockingPipeline
import android.media.AudioFormat as AndroidAudioFormat
import app.earshield.core.AudioFormat as Fmt

/**
 * Runs the passthrough loop on a dedicated audio thread:
 * mic → [BlockingPipeline] → earbuds.
 *
 * Phone mic mode: records from the phone's built-in mic and plays over normal Bluetooth audio,
 * so you keep full sound quality.
 *
 * Earbud mic mode: switches Bluetooth into headset (call) mode so the earbuds' mic can be used.
 * Bluetooth only carries one mono mic stream in this mode, and what you hear also drops to
 * call quality. Android's echo cancellation is on in this mode, which stops the earbud speaker
 * from feeding back into the earbud mic.
 *
 * Privacy guard: audio is only ever played to earbuds or headphones. If Android routes output
 * anywhere else (for example the phone speaker when earbuds disconnect or die), the output is
 * silenced at once and stays silent until private output is back. Listening and blocking continue.
 */
class AudioEngine(
    private val context: Context,
    private val pipeline: BlockingPipeline,
    private val mode: InputMode,
    private val onError: (String) -> Unit,
    /** Called with true when the mic has delivered nothing but zeros for 2 s, false when sound returns. */
    private val onSilentMic: (Boolean) -> Unit = {},
    /** Called with false when output was silenced because it isn't going to earbuds/headphones. */
    private val onPrivateOutput: (Boolean) -> Unit = {},
    /** About once a second: where sound is going, and how many playback gaps (underruns) so far. */
    private val onOutputInfo: (route: String, gaps: Int) -> Unit = { _, _ -> },
) {
    /** True only while output is confirmed to be going to earbuds or headphones. */
    @Volatile private var outputPrivate = false
    /** After an "unplugged" alert, don't trust the route again until this time (elapsed ms). */
    @Volatile private var distrustUntil = 0L
    @Volatile private var activeTrack: AudioTrack? = null
    @Volatile private var preview: FloatArray? = null
    @Volatile private var previewPos = 0

    /** Plays [clip] to the earbuds over the (turned-down) room sound. Only ever to private output. */
    fun playPreview(clip: FloatArray) {
        previewPos = 0
        preview = clip
    }
    /** Set when queued audio was thrown away; playback restarts once output is private again. */
    @Volatile private var needsRestart = false

    /** Silences output immediately, discarding the ~250 ms of audio already queued for playback. */
    private fun silenceNow() {
        outputPrivate = false
        activeTrack?.let { t ->
            runCatching { t.pause(); t.flush() }
            needsRestart = true
        }
    }

    /** Fires the instant headphones are unplugged or Bluetooth audio drops, before Android reroutes. */
    private val noisyReceiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context, intent: Intent) {
            if (intent.action == AudioManager.ACTION_AUDIO_BECOMING_NOISY) {
                distrustUntil = SystemClock.elapsedRealtime() + 1500
                silenceNow()
            }
        }
    }

    private val audioManager = context.getSystemService(AudioManager::class.java)
    @Volatile private var running = false
    private var thread: Thread? = null

    fun start() {
        running = true
        thread = Thread(::run, "earshield-audio").also { it.start() }
    }

    fun stop() {
        running = false
        thread?.join(3000)
        thread = null
    }

    @SuppressLint("MissingPermission") // RECORD_AUDIO is checked before the service starts.
    private fun run() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
        var record: AudioRecord? = null
        var track: AudioTrack? = null
        var headset: AudioDeviceInfo? = null
        try {
            if (mode == InputMode.EARBUD_MIC) headset = routeToHeadset()
            record = buildRecord(headset)
            track = buildTrack(headset)
            check(record.state == AudioRecord.STATE_INITIALIZED) { "Couldn't open the microphone." }
            check(track.state == AudioTrack.STATE_INITIALIZED) { "Couldn't open audio output." }
            ContextCompat.registerReceiver(context, noisyReceiver,
                IntentFilter(AudioManager.ACTION_AUDIO_BECOMING_NOISY), ContextCompat.RECEIVER_NOT_EXPORTED)
            val routedTrack = track
            activeTrack = track
            routedTrack.addOnRoutingChangedListener({ checkRoute(routedTrack) }, null)
            record.startRecording()
            track.play()
            checkRoute(track)
            var reportedPrivate: Boolean? = null
            var chunkCount = 0

            val n = Fmt.CHUNK
            val pcmIn = ShortArray(n)
            val pcmOut = ShortArray(n)
            val inF = FloatArray(n)
            val outF = FloatArray(n)
            var zeroChunks = 0
            var reportedSilent = false
            while (running) {
                var got = 0
                while (got < n && running) {
                    val r = record.read(pcmIn, got, n - got, AudioRecord.READ_BLOCKING)
                    check(r >= 0) { "Microphone stopped working (error $r)." }
                    got += r
                }
                if (!running) break
                // Android feeds zeros when mic access is blocked or another app holds the mic.
                if (pcmIn.all { it.toInt() == 0 }) zeroChunks++ else zeroChunks = 0
                val silent = zeroChunks * n >= Fmt.SAMPLE_RATE * 2
                if (silent != reportedSilent) {
                    reportedSilent = silent
                    onSilentMic(silent)
                }
                for (i in 0 until n) inF[i] = pcmIn[i] / 32768f
                pipeline.process(inF, outF)
                preview?.let { pv ->
                    val start = previewPos
                    for (i in 0 until n) outF[i] = outF[i] * 0.15f + pv.getOrElse(start + i) { 0f }
                    previewPos = start + n
                    if (previewPos >= pv.size) preview = null
                }
                if (++chunkCount % 8 == 0) checkRoute(track) // re-check about every 250 ms as a backstop
                if (chunkCount % 32 == 0) onOutputInfo(lastRoute, track.underrunCount)
                val private = outputPrivate
                if (private && needsRestart) {
                    needsRestart = false
                    track.play()
                }
                if (private != reportedPrivate) {
                    reportedPrivate = private
                    onPrivateOutput(private)
                }
                if (private) {
                    for (i in 0 until n) pcmOut[i] = (outF[i].coerceIn(-1f, 1f) * 32767f).toInt().toShort()
                } else {
                    pcmOut.fill(0) // never play the room through the phone
                }
                // Non-blocking: if the output clock runs slightly slow, drop rather than build up delay.
                track.write(pcmOut, 0, n, AudioTrack.WRITE_NON_BLOCKING)
            }
        } catch (e: Exception) {
            Log.e(TAG, "audio loop failed", e)
            onError(e.message ?: e.toString())
        } finally {
            outputPrivate = false
            activeTrack = null
            runCatching { context.unregisterReceiver(noisyReceiver) }
            runCatching { record?.stop() }
            record?.release()
            runCatching { track?.stop() }
            track?.release()
            if (mode == InputMode.EARBUD_MIC) restoreRouting()
        }
    }

    /**
     * Allows output only once the track is confirmed to be playing to earbuds or headphones, and
     * silences it the moment it is confirmed to be going anywhere else. Android sometimes briefly
     * can't say where a track is routed; that keeps the current state instead of cutting the sound.
     */
    private fun checkRoute(track: AudioTrack) {
        val device = track.routedDevice
        lastRoute = device?.let { describe(it.type) } ?: "unknown"
        if (device == null) return
        val ok = device.type in PRIVATE_OUTPUTS && SystemClock.elapsedRealtime() >= distrustUntil
        if (!ok && outputPrivate) silenceNow() else outputPrivate = ok
    }

    @Volatile private var lastRoute = "unknown"

    private fun describe(type: Int): String = when (type) {
        AudioDeviceInfo.TYPE_BLUETOOTH_A2DP -> "Bluetooth"
        AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> "Bluetooth (call mode)"
        AudioDeviceInfo.TYPE_WIRED_HEADSET, AudioDeviceInfo.TYPE_WIRED_HEADPHONES -> "Wired headphones"
        AudioDeviceInfo.TYPE_USB_HEADSET -> "USB headphones"
        AudioDeviceInfo.TYPE_BUILTIN_SPEAKER -> "Phone speaker (blocked)"
        AudioDeviceInfo.TYPE_BUILTIN_EARPIECE -> "Phone earpiece (blocked)"
        else -> if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && type == AudioDeviceInfo.TYPE_BLE_HEADSET) "Bluetooth LE"
        else if (type in PRIVATE_OUTPUTS) "Headphones" else "Other output (blocked)"
    }

    /** Switches Bluetooth to headset mode and returns the earbud mic device. */
    private fun routeToHeadset(): AudioDeviceInfo? {
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val devices = audioManager.availableCommunicationDevices
            val device = devices.firstOrNull { it.type == AudioDeviceInfo.TYPE_BLE_HEADSET }
                ?: devices.firstOrNull { it.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO }
                ?: error("No Bluetooth earbud mic found. Connect your earbuds, or switch to the phone mic.")
            check(audioManager.setCommunicationDevice(device)) { "Android wouldn't switch to the earbud mic." }
            return device
        }
        @Suppress("DEPRECATION")
        run {
            audioManager.startBluetoothSco()
            audioManager.isBluetoothScoOn = true
        }
        val deadline = SystemClock.elapsedRealtime() + 5000
        while (SystemClock.elapsedRealtime() < deadline) {
            val sticky = context.registerReceiver(null, IntentFilter(AudioManager.ACTION_SCO_AUDIO_STATE_UPDATED))
            val state = sticky?.getIntExtra(AudioManager.EXTRA_SCO_AUDIO_STATE, -1)
            if (state == AudioManager.SCO_AUDIO_STATE_CONNECTED) {
                return audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)
                    .firstOrNull { it.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO }
            }
            Thread.sleep(100)
        }
        error("The earbud mic didn't connect. Make sure your earbuds are connected, or switch to the phone mic.")
    }

    private fun restoreRouting() {
        runCatching {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                audioManager.clearCommunicationDevice()
            } else {
                @Suppress("DEPRECATION")
                audioManager.isBluetoothScoOn = false
                @Suppress("DEPRECATION")
                audioManager.stopBluetoothSco()
            }
            audioManager.mode = AudioManager.MODE_NORMAL
        }
    }

    @SuppressLint("MissingPermission")
    private fun buildRecord(headset: AudioDeviceInfo?): AudioRecord {
        // VOICE_RECOGNITION is tuned for speech and works on every phone. (UNPROCESSED was tried
        // first, but on many phones it delivers speech too quiet to detect.)
        val source = if (mode == InputMode.EARBUD_MIC) MediaRecorder.AudioSource.VOICE_COMMUNICATION
        else MediaRecorder.AudioSource.VOICE_RECOGNITION
        val format = AndroidAudioFormat.Builder()
            .setSampleRate(Fmt.SAMPLE_RATE)
            .setChannelMask(AndroidAudioFormat.CHANNEL_IN_MONO)
            .setEncoding(AndroidAudioFormat.ENCODING_PCM_16BIT)
            .build()
        val minBuf = AudioRecord.getMinBufferSize(
            Fmt.SAMPLE_RATE, AndroidAudioFormat.CHANNEL_IN_MONO, AndroidAudioFormat.ENCODING_PCM_16BIT)
        val record = AudioRecord.Builder()
            .setAudioSource(source)
            .setAudioFormat(format)
            .setBufferSizeInBytes(maxOf(minBuf, Fmt.SAMPLE_RATE * 2)) // 1 s of slack
            .build()
        val preferred = headset ?: audioManager.getDevices(AudioManager.GET_DEVICES_INPUTS)
            .firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_MIC }
        preferred?.let { record.setPreferredDevice(it) }
        return record
    }

    private fun buildTrack(headset: AudioDeviceInfo?): AudioTrack {
        val attrs = AudioAttributes.Builder().apply {
            if (mode == InputMode.EARBUD_MIC) {
                setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            } else {
                setUsage(AudioAttributes.USAGE_MEDIA)
                setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
            }
        }.build()
        val format = AndroidAudioFormat.Builder()
            .setSampleRate(Fmt.SAMPLE_RATE)
            .setChannelMask(AndroidAudioFormat.CHANNEL_OUT_MONO)
            .setEncoding(AndroidAudioFormat.ENCODING_PCM_16BIT)
            .build()
        val minBuf = AudioTrack.getMinBufferSize(
            Fmt.SAMPLE_RATE, AndroidAudioFormat.CHANNEL_OUT_MONO, AndroidAudioFormat.ENCODING_PCM_16BIT)
        val track = AudioTrack.Builder()
            .setAudioAttributes(attrs)
            .setAudioFormat(format)
            .setBufferSizeInBytes(maxOf(minBuf, Fmt.SAMPLE_RATE / 2)) // ~250 ms
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
            .build()
        // Earbud mic mode: play to the call-mode side of the same earbuds. Phone mic mode: let Android
        // route media normally. (Steering it ourselves could pick the earbuds' call-mode output,
        // which isn't open in this mode and made sound drop in and out.)
        if (headset != null) {
            audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
                .firstOrNull { it.type == headset.type }
                ?.let { track.setPreferredDevice(it) }
        }
        return track
    }

    private companion object {
        const val TAG = "AudioEngine"

        /** Outputs only the wearer can hear. Speakers (phone, Bluetooth, car, TV…) are never allowed. */
        val PRIVATE_OUTPUTS: Set<Int> = buildSet {
            add(AudioDeviceInfo.TYPE_BLUETOOTH_A2DP)
            add(AudioDeviceInfo.TYPE_BLUETOOTH_SCO)
            add(AudioDeviceInfo.TYPE_WIRED_HEADSET)
            add(AudioDeviceInfo.TYPE_WIRED_HEADPHONES)
            add(AudioDeviceInfo.TYPE_USB_HEADSET)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) add(AudioDeviceInfo.TYPE_HEARING_AID)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) add(AudioDeviceInfo.TYPE_BLE_HEADSET)
        }
    }
}
