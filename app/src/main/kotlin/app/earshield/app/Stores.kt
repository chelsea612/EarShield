package app.earshield.app

import android.content.Context
import app.earshield.core.BlockingSettings
import app.earshield.core.CoverStyle
import app.earshield.core.OverlapStyle
import app.earshield.core.Vectors
import app.earshield.core.VoiceProfile
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

enum class InputMode { PHONE_MIC, EARBUD_MIC }

data class AppSettings(
    val inputMode: InputMode = InputMode.PHONE_MIC,
    val delaySeconds: Float = BlockingSettings().delaySeconds,
    val threshold: Float = BlockingSettings().threshold,
    val cover: CoverStyle = BlockingSettings().cover,
    val overlap: OverlapStyle = BlockingSettings().overlap,
    val blockStrangers: Boolean = false,
    val buzzOnBlock: Boolean = true,
)

class SettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val _settings = MutableStateFlow(load())
    val settings: StateFlow<AppSettings> = _settings.asStateFlow()

    private fun load(): AppSettings {
        val d = AppSettings()
        // Version 0.3 changed what the delay and threshold mean; start those from the new defaults.
        if (prefs.getInt("settingsVersion", 1) < SETTINGS_VERSION) {
            prefs.edit().remove("delaySeconds").remove("threshold").putInt("settingsVersion", SETTINGS_VERSION).apply()
        }
        return AppSettings(
            inputMode = enumOr(prefs.getString("inputMode", null), d.inputMode),
            delaySeconds = prefs.getFloat("delaySeconds", d.delaySeconds),
            threshold = prefs.getFloat("threshold", d.threshold),
            cover = enumOr(prefs.getString("cover", null), d.cover),
            overlap = enumOr(prefs.getString("overlap", null), d.overlap),
            blockStrangers = prefs.getBoolean("blockStrangers", d.blockStrangers),
            buzzOnBlock = prefs.getBoolean("buzzOnBlock", d.buzzOnBlock),
        )
    }

    fun update(transform: (AppSettings) -> AppSettings) {
        val s = transform(_settings.value)
        _settings.value = s
        prefs.edit()
            .putString("inputMode", s.inputMode.name)
            .putFloat("delaySeconds", s.delaySeconds)
            .putFloat("threshold", s.threshold)
            .putString("cover", s.cover.name)
            .putString("overlap", s.overlap.name)
            .putBoolean("blockStrangers", s.blockStrangers)
            .putBoolean("buzzOnBlock", s.buzzOnBlock)
            .apply()
    }

    private companion object {
        const val SETTINGS_VERSION = 2
    }

    private inline fun <reified T : Enum<T>> enumOr(name: String?, default: T): T =
        name?.let { runCatching { enumValueOf<T>(it) }.getOrNull() } ?: default
}

/** Saves voiceprints (never audio) to a private file on the phone. */
class VoiceStore(context: Context) {
    private companion object {
        /** Voiceprints only work with the model that made them. */
        const val MODEL = "wespeaker-resnet34"
    }

    private val file = File(context.filesDir, "voices.json")
    private val _voices = MutableStateFlow(load())
    val voices: StateFlow<List<VoiceProfile>> = _voices.asStateFlow()

    /** Names of voices recorded with the old voice model, which had to be dropped. */
    var dropped: List<String> = emptyList()
        private set

    private fun load(): List<VoiceProfile> = runCatching {
        val arr = JSONArray(file.readText())
        val all = (0 until arr.length()).map { arr.getJSONObject(it) }
        val (current, old) = all.partition { it.optString("model") == MODEL }
        dropped = old.map { it.getString("name") }.filterNot { it.startsWith("Not ") }
        current.map { o ->
            val e = o.getJSONArray("embedding")
            VoiceProfile(
                id = o.getString("id"),
                name = o.getString("name"),
                embedding = FloatArray(e.length()) { e.getDouble(it).toFloat() },
                createdAtMillis = o.optLong("createdAt"),
                allowed = o.optBoolean("allowed", false),
            )
        }
    }.getOrDefault(emptyList()).also { if (dropped.isNotEmpty()) save(it) }

    @Synchronized
    fun add(name: String, embedding: FloatArray, allowed: Boolean = false): VoiceProfile {
        val v = VoiceProfile(UUID.randomUUID().toString(), name, embedding, allowed = allowed)
        save(_voices.value + v)
        return v
    }

    @Synchronized
    fun remove(id: String) = save(_voices.value.filterNot { it.id == id })

    /** Renames a voice and/or moves it between Blocked and Never blocked. */
    @Synchronized
    fun update(id: String, name: String, allowed: Boolean) = save(_voices.value.map { v ->
        if (v.id == id) VoiceProfile(v.id, name.trim().ifEmpty { v.name }, v.embedding, v.createdAtMillis, allowed) else v
    })

    /**
     * Combines two saved voices of the same person into one: the voiceprints are averaged, and the
     * result keeps [keepId]'s place, with the given name and list.
     */
    @Synchronized
    fun merge(keepId: String, otherId: String, name: String, allowed: Boolean) {
        val keep = _voices.value.firstOrNull { it.id == keepId } ?: return
        val other = _voices.value.firstOrNull { it.id == otherId } ?: return
        val combined = Vectors.normalized(Vectors.mean(listOf(keep.embedding, other.embedding)))
        save(_voices.value.mapNotNull { v ->
            when (v.id) {
                keepId -> VoiceProfile(v.id, name.trim().ifEmpty { v.name }, combined, minOf(keep.createdAtMillis, other.createdAtMillis), allowed)
                otherId -> null
                else -> v
            }
        })
    }

    private fun save(list: List<VoiceProfile>) {
        val arr = JSONArray()
        for (v in list) {
            arr.put(JSONObject().apply {
                put("id", v.id)
                put("name", v.name)
                put("createdAt", v.createdAtMillis)
                put("allowed", v.allowed)
                put("model", MODEL)
                put("embedding", JSONArray().apply { v.embedding.forEach { put(it.toDouble()) } })
            })
        }
        val tmp = File(file.parentFile, "voices.json.tmp")
        tmp.writeText(arr.toString())
        tmp.renameTo(file)
        _voices.value = list
    }
}
