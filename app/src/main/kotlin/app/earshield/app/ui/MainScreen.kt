package app.earshield.app.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.foundation.clickable
import app.earshield.core.Vectors
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import app.earshield.app.AppSettings
import app.earshield.app.InputMode
import app.earshield.app.LiveStatus
import app.earshield.app.SaveOutcome
import app.earshield.app.VoiceRole
import app.earshield.app.LiveRow
import app.earshield.core.LiveSpeaker
import app.earshield.app.DiscoveredVoice
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import app.earshield.core.VoiceDiscovery
import app.earshield.app.ShieldController
import app.earshield.core.CoverStyle
import app.earshield.core.OverlapStyle
import app.earshield.core.Enrollment
import app.earshield.core.VoiceProfile
import kotlinx.coroutines.launch
import java.text.DateFormat
import java.util.Date
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(controller: ShieldController) {
    val status by controller.status.collectAsStateWithLifecycle()
    val settings by controller.settingsStore.settings.collectAsStateWithLifecycle()
    val voices by controller.voiceStore.voices.collectAsStateWithLifecycle()
    val context = LocalContext.current
    // null = hidden; BLOCK = add a voice to block; ME = teach your voice; ALLOW = allow someone else
    var addMode by remember { mutableStateOf<VoiceRole?>(null) }
    var learning by remember { mutableStateOf(false) }
    var permissionNote by remember { mutableStateOf<String?>(null) }
    var droppedSeen by remember { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        if (results[Manifest.permission.RECORD_AUDIO] == true || hasMic(context)) {
            permissionNote = null
            controller.start()
        } else {
            permissionNote = "Earshield needs microphone access to hear the room."
        }
    }

    fun startWithPermissions() {
        val needed = buildList {
            add(Manifest.permission.RECORD_AUDIO)
            if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.POST_NOTIFICATIONS)
            if (Build.VERSION.SDK_INT >= 31 && settings.inputMode == InputMode.EARBUD_MIC) {
                add(Manifest.permission.BLUETOOTH_CONNECT)
            }
        }.filter { ContextCompat.checkSelfPermission(context, it) != PackageManager.PERMISSION_GRANTED }
        if (needed.isEmpty()) controller.start() else permissionLauncher.launch(needed.toTypedArray())
    }

    Scaffold(
        topBar = { TopAppBar(title = { Text("Earshield", fontWeight = FontWeight.SemiBold) }) },
        containerColor = MaterialTheme.colorScheme.background,
    ) { padding ->
        LazyColumn(
            contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = padding.calculateTopPadding(),
                bottom = padding.calculateBottomPadding() + 24.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item {
                StatusCard(status, settings, voices.isEmpty(),
                    onStart = ::startWithPermissions, onStop = controller::stop, onPause = controller::setPaused)
            }
            if (status.running && status.outputMuted) {
                item {
                    ErrorCard("Sound is paused: no earbuds or headphones are connected. Earshield never plays " +
                        "the room through your phone's speaker. Still listening and blocking.") {}
                }
            }
            if (status.micSilent) {
                item {
                    ErrorCard("The mic is sending complete silence. Check that microphone access is on " +
                        "(Quick Settings) and that no call or other app is using the mic.") {}
                }
            }
            if (status.running) item { WhoIsTalkingCard(status, controller) }
            (status.error ?: permissionNote)?.let { msg ->
                item { ErrorCard(msg) { controller.clearError(); permissionNote = null } }
            }
            if (controller.voiceStore.dropped.isNotEmpty() && !droppedSeen) {
                item {
                    ErrorCard("This version recognizes voices with a new, more accurate model. Please record again: " +
                        controller.voiceStore.dropped.joinToString(", ") + ".") { droppedSeen = true }
                }
            }
            item {
                VoicesCard(voices, controller, onAdd = { addMode = when (it) { false -> VoiceRole.BLOCK; true -> VoiceRole.ME; null -> VoiceRole.ALLOW } }, onLearn = { learning = true })
            }
            item { MicCard(settings, status.running) { mode -> controller.settingsStore.update { it.copy(inputMode = mode) } } }
            item { BlockingCard(settings) { change -> controller.settingsStore.update(change) } }
            item { SoundCard(settings) { change -> controller.settingsStore.update(change) } }
            if (status.recentBlocks.isNotEmpty()) item { RecentCard(status, controller::markNotThem) }
            item {
                Text(
                    "Voiceprints stay on this phone. Earshield doesn't save or send any audio.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 4.dp),
                )
            }
        }
    }

    if (learning) {
        LearnVoicesDialog(controller, status, onStartListening = ::startWithPermissions, onDismiss = { learning = false })
    }
    addMode?.let { role ->
        AddVoiceDialog(controller, status, role, onStartListening = ::startWithPermissions, onDismiss = { addMode = null })
    }
}

private fun hasMic(context: android.content.Context) =
    ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

@Composable
private fun SectionCard(content: @Composable () -> Unit) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) { content() }
    }
}

@Composable
private fun SectionTitle(text: String) =
    Text(text, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)

@Composable
private fun Hint(text: String) =
    Text(text, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)

@Composable
private fun StatusCard(
    status: LiveStatus,
    settings: AppSettings,
    noVoices: Boolean,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onPause: (Boolean) -> Unit,
) {
    val blocking = status.blockingName != null && !status.paused
    val dotColor by animateColorAsState(
        when {
            !status.running -> MaterialTheme.colorScheme.outline
            blocking -> TransPink
            status.speaking -> TransBlue
            else -> Color(0xFF7FD69B)
        }, label = "dot")
    SectionCard {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(14.dp).background(dotColor, CircleShape))
            Spacer(Modifier.width(10.dp))
            Text(
                when {
                    status.starting -> "Starting…"
                    !status.running -> "Off"
                    status.paused -> "Paused: you hear everything"
                    blocking -> "Covering ${status.blockingName}"
                    status.speaking -> "Someone is talking"
                    else -> "Listening"
                },
                style = MaterialTheme.typography.headlineSmall,
            )
        }
        if (status.running) {
            val mic = if (settings.inputMode == InputMode.PHONE_MIC) "Phone mic" else "Earbud mic"
            Hint("$mic · you hear the room ${"%.1f".format(settings.delaySeconds)} s late")
            if (System.currentTimeMillis() - status.fellBehindAtMillis < 10_000) {
                Text("Your phone couldn't keep up, so some audio played unchecked. Try a longer delay, " +
                    "or set overlaps to \"Turn down\".", color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall)
            }
            MicMeter(status)
            if (status.outputRoute.isNotEmpty()) {
                Hint("Sound to: ${status.outputRoute} · playback gaps: ${status.playbackGaps}")
            }
            val name = status.closestName
            if (status.speaking && name != null) {
                Text("Closest blocked voice: $name (${(status.closestScore * 100).roundToInt()}% match, blocks at ${(settings.threshold * 100).roundToInt()}%)",
                    style = MaterialTheme.typography.bodyMedium)
                LinearProgressIndicator(
                    progress = { status.closestScore.coerceIn(0f, 1f) },
                    color = if (status.closestScore >= settings.threshold) TransPink else TransBlue,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        } else if (noVoices) {
            Hint("Start listening, then add the voices you want blocked.")
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (status.running || status.starting) {
                Button(onClick = onStop) { Text("Stop") }
                OutlinedButton(onClick = { onPause(!status.paused) }, enabled = status.running) {
                    Text(if (status.paused) "Resume blocking" else "Pause blocking")
                }
            } else {
                Button(onClick = onStart) { Text("Start listening") }
            }
        }
    }
}

/** Live mic level, so you can see whether Earshield hears anything. */
@Composable
private fun MicMeter(status: LiveStatus) {
    // Map −70 dBFS (near silence) … −10 dBFS (loud) onto the bar.
    val fraction = ((status.micDb + 70f) / 60f).coerceIn(0f, 1f)
    val isSpeech = status.speechProbability >= 0.5f
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row {
            Text("Mic level", style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
            Text(
                if (status.micSilent) "no sound" else if (isSpeech) "speech" else "%.0f dB".format(status.micDb),
                style = MaterialTheme.typography.bodySmall,
                color = if (isSpeech) TransPink else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        LinearProgressIndicator(
            progress = { fraction },
            color = if (isSpeech) TransPink else TransBlue,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun ErrorCard(message: String, onDismiss: () -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
        modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Filled.Warning, contentDescription = null, tint = MaterialTheme.colorScheme.onErrorContainer)
            Spacer(Modifier.width(12.dp))
            Text(message, color = MaterialTheme.colorScheme.onErrorContainer, modifier = Modifier.weight(1f))
            TextButton(onClick = onDismiss) { Text("OK") }
        }
    }
}

@Composable
private fun VoicesCard(
    voices: List<VoiceProfile>,
    controller: ShieldController,
    onAdd: (allowed: Boolean?) -> Unit,
    onLearn: () -> Unit,
) {
    var editing by remember { mutableStateOf<VoiceProfile?>(null) }
    val blocked = voices.filterNot { it.allowed }
    val allowed = voices.filter { it.allowed }
    val dates = remember { DateFormat.getDateInstance(DateFormat.MEDIUM) }

    @Composable
    fun VoiceRow(v: VoiceProfile) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth().clickable { editing = v }.padding(vertical = 4.dp),
        ) {
            Column(Modifier.weight(1f)) {
                Text(v.name, style = MaterialTheme.typography.bodyLarge)
                Hint("Added ${dates.format(Date(v.createdAtMillis))} · tap to move, rename or merge")
            }
            // One-tap move to the other list.
            TextButton(onClick = { controller.updateVoice(v.id, v.name, !v.allowed) }) {
                Text(if (v.allowed) "Block" else "Allow")
            }
            Icon(Icons.Filled.Edit, contentDescription = "Edit ${v.name}", tint = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }

    SectionCard {
        SectionTitle("Learn voices")
        Hint("Record a normal conversation, even with people talking over each other. Earshield finds each " +
            "person and lets you listen to them and choose who to block or allow.")
        Button(onClick = onLearn) { Text("Learn voices from a conversation") }
    }
    Spacer(Modifier.height(12.dp))
    SectionCard {
        SectionTitle("Blocked voices")
        if (blocked.isEmpty()) Hint("No one yet.")
        blocked.forEachIndexed { i, v ->
            if (i > 0) HorizontalDivider()
            VoiceRow(v)
        }
        FilledTonalButton(onClick = { onAdd(false) }) {
            Icon(Icons.Filled.Add, contentDescription = null, modifier = Modifier.size(18.dp))
            Spacer(Modifier.width(6.dp))
            Text("Add one voice")
        }
    }
    Spacer(Modifier.height(12.dp))
    SectionCard {
        SectionTitle("Never blocked")
        Hint("You, friends, and anything you marked \"Not them\". A voice is only covered if it sounds clearly " +
            "more like a blocked person than like anything here.")
        allowed.forEachIndexed { i, v ->
            if (i > 0) HorizontalDivider()
            VoiceRow(v)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (allowed.none { it.name == "You" }) {
                OutlinedButton(onClick = { onAdd(true) }) { Text("Teach it my voice") }
            }
            OutlinedButton(onClick = { onAdd(null) }) { Text("Allow one voice") }
        }
    }
    editing?.let { v -> EditVoiceDialog(v, voices, controller, onDismiss = { editing = null }) }
}

/** Rename a voice, move it between lists, merge it with another saved voice, or delete it. */
@Composable
private fun EditVoiceDialog(
    voice: VoiceProfile,
    voices: List<VoiceProfile>,
    controller: ShieldController,
    onDismiss: () -> Unit,
) {
    var name by remember { mutableStateOf(voice.name) }
    var allowed by remember { mutableStateOf(voice.allowed) }
    var mergeWith by remember { mutableStateOf<VoiceProfile?>(null) }
    var confirmDelete by remember { mutableStateOf(false) }
    val others = voices.filter { it.id != voice.id }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Delete ${voice.name}?") },
            text = { Text("This voiceprint will be deleted from this phone.") },
            confirmButton = { TextButton(onClick = { controller.removeVoice(voice.id); onDismiss() }) { Text("Delete") } },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } },
        )
        return
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Edit voice") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("Name") },
                    singleLine = true, modifier = Modifier.fillMaxWidth())

                Text("List", style = MaterialTheme.typography.bodyMedium)
                SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                    SegmentedButton(selected = !allowed, onClick = { allowed = false },
                        shape = SegmentedButtonDefaults.itemShape(0, 2)) { Text("Blocked") }
                    SegmentedButton(selected = allowed, onClick = { allowed = true },
                        shape = SegmentedButtonDefaults.itemShape(1, 2)) { Text("Never blocked") }
                }

                if (others.isNotEmpty()) {
                    Text("Merge with another voice", style = MaterialTheme.typography.bodyMedium)
                    Hint("If the same person was saved twice, combine them into one, more reliable voice.")
                    others.forEach { o ->
                        val sim = Vectors.dot(voice.embedding, o.embedding)
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth().clickable { mergeWith = if (mergeWith?.id == o.id) null else o },
                        ) {
                            androidx.compose.material3.RadioButton(selected = mergeWith?.id == o.id, onClick = {
                                mergeWith = if (mergeWith?.id == o.id) null else o
                            })
                            Column(Modifier.weight(1f)) {
                                Text(o.name)
                                Hint((if (o.allowed) "Never blocked" else "Blocked") + " · ${(sim * 100).toInt()}% alike")
                            }
                        }
                    }
                    mergeWith?.let { o ->
                        if (Vectors.dot(voice.embedding, o.embedding) < MERGE_WARN) {
                            Text("These two sound like different people. Merging them could block or allow the wrong voices.",
                                color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }

                TextButton(onClick = { confirmDelete = true }) {
                    Icon(Icons.Filled.Delete, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Delete this voice", color = MaterialTheme.colorScheme.error)
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                val m = mergeWith
                if (m != null) controller.mergeVoices(voice.id, m.id, name, allowed)
                else controller.updateVoice(voice.id, name, allowed)
                onDismiss()
            }) { Text(if (mergeWith != null) "Merge and save" else "Save") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

/** Below this similarity, merging two voices probably combines two different people. */
private const val MERGE_WARN = 0.40f

/** Noise reduction, distant-chatter reduction, and a masking noise bed. */
@Composable
private fun SoundCard(settings: AppSettings, onChange: ((AppSettings) -> AppSettings) -> Unit) {
    SectionCard {
        SectionTitle("Sound")
        fun pct(v: Float) = if (v <= 0f) "Off" else "${(v * 100).roundToInt()}%"
        LabeledSlider("Background noise reduction", settings.noiseReduction, pct(settings.noiseReduction), 0f..1f, 9) { v ->
            onChange { it.copy(noiseReduction = (v * 10).roundToInt() / 10f) }
        }
        Hint("Turns down traffic, hum and hiss while keeping voices. At full strength some voices get a little quieter too.")
        LabeledSlider("Background chatter", settings.chatterReduction, pct(settings.chatterReduction), 0f..1f, 9) { v ->
            onChange { it.copy(chatterReduction = (v * 10).roundToInt() / 10f) }
        }
        Hint("Turns down distant voices that are much quieter than your conversation, in the pauses between turns. " +
            "Works best after \"Teach it my voice\", since your own voice is the loudest the mic hears.")
        LabeledSlider("Masking noise (brown)", settings.maskingNoise, pct(settings.maskingNoise), 0f..1f, 9) { v ->
            onChange { it.copy(maskingNoise = (v * 10).roundToInt() / 10f) }
        }
        Hint("A steady, deep rumble under everything, to hide sound leaking in around your earbuds.")
    }
}

@Composable
private fun MicCard(settings: AppSettings, running: Boolean, onChange: (InputMode) -> Unit) {
    SectionCard {
        SectionTitle("Microphone")
        val options = listOf(InputMode.PHONE_MIC to "Phone mic", InputMode.EARBUD_MIC to "Earbud mic")
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            options.forEachIndexed { i, (mode, label) ->
                SegmentedButton(
                    selected = settings.inputMode == mode,
                    onClick = { onChange(mode) },
                    shape = SegmentedButtonDefaults.itemShape(i, options.size),
                ) { Text(label) }
            }
        }
        Hint(
            if (settings.inputMode == InputMode.PHONE_MIC)
                "Full sound quality. Works best with the phone in a shirt pocket or on a lanyard, not a trouser pocket."
            else
                "Uses your earbuds' mics. Bluetooth switches to call mode: the mic is mono, and what you hear drops to phone-call quality while Earshield is on."
        )
        if (running) Hint("Switching restarts listening.")
    }
}

@Composable
private fun BlockingCard(settings: AppSettings, onChange: ((AppSettings) -> AppSettings) -> Unit) {
    SectionCard {
        SectionTitle("Blocking")

        LabeledSlider(
            label = "Delay",
            value = settings.delaySeconds,
            valueText = "%.1f s".format(settings.delaySeconds),
            range = 1f..5f,
            steps = 39,
        ) { v -> onChange { it.copy(delaySeconds = (v * 10).roundToInt() / 10f) } }
        Hint(
            if (settings.delaySeconds < 2.5f)
                "Short delays feel more natural, but the first words of a blocked person can get through."
            else
                "Enough time to catch a blocked voice before you hear it. Lips and voices won't quite line up."
        )

        // Shown as strictness: right = stricter = higher match needed.
        LabeledSlider(
            label = "Match needed",
            value = settings.threshold,
            valueText = "${(settings.threshold * 100).roundToInt()}%",
            range = 0.25f..0.70f,
            steps = 44,
        ) { v -> onChange { it.copy(threshold = (v * 100).roundToInt() / 100f) } }
        Hint("Lower blocks more reliably but may cover people who sound similar. Raise it if friends get covered.")

        Text("When they talk over someone", style = MaterialTheme.typography.bodyMedium)
        val overlaps = listOf(OverlapStyle.SEPARATE to "Separate", OverlapStyle.TURN_DOWN to "Turn down", OverlapStyle.COVER to "Cover")
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            overlaps.forEachIndexed { i, (style, label) ->
                SegmentedButton(
                    selected = settings.overlap == style,
                    onClick = { onChange { it.copy(overlap = style) } },
                    shape = SegmentedButtonDefaults.itemShape(i, overlaps.size),
                ) { Text(label) }
            }
        }
        Hint(when (settings.overlap) {
            OverlapStyle.SEPARATE -> "Pulls their voice out of the mix so you still hear the other person. Uses more battery. " +
                "If the voices can't be told apart, they're turned down instead."
            OverlapStyle.TURN_DOWN -> "Both voices are turned down while they overlap."
            OverlapStyle.COVER -> "Both voices are covered while they overlap."
        })

        Text("Cover blocked speech with", style = MaterialTheme.typography.bodyMedium)
        val covers = listOf(CoverStyle.NATURAL to "Natural", CoverStyle.SILENCE to "Silence", CoverStyle.QUIETER to "Quieter", CoverStyle.SOFT_NOISE to "Hush")
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
            covers.forEachIndexed { i, (style, label) ->
                SegmentedButton(
                    selected = settings.cover == style,
                    onClick = { onChange { it.copy(cover = style) } },
                    shape = SegmentedButtonDefaults.itemShape(i, covers.size),
                ) { Text(label) }
            }
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Block everyone I haven't allowed", style = MaterialTheme.typography.bodyMedium)
                Hint(if (settings.blockStrangers) "Only you and your allowed voices are heard. Anyone else is covered."
                    else "Off: only blocked voices are covered.")
            }
            Switch(checked = settings.blockStrangers, onCheckedChange = { b -> onChange { it.copy(blockStrangers = b) } })
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Buzz when blocking", style = MaterialTheme.typography.bodyMedium)
                Hint("So you know something was said.")
            }
            Switch(checked = settings.buzzOnBlock, onCheckedChange = { b -> onChange { it.copy(buzzOnBlock = b) } })
        }
    }
}

@Composable
private fun LabeledSlider(
    label: String,
    value: Float,
    valueText: String,
    range: ClosedFloatingPointRange<Float>,
    steps: Int,
    onChange: (Float) -> Unit,
) {
    Column {
        Row {
            Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
            Text(valueText, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
        }
        Slider(value = value, onValueChange = onChange, valueRange = range, steps = steps)
    }
}

@Composable
private fun RecentCard(status: LiveStatus, onNotThem: (Long) -> Unit) {
    val fmt = remember { DateFormat.getTimeInstance(DateFormat.SHORT) }
    SectionCard {
        SectionTitle("Recently covered")
        Hint("Covered someone by mistake? Tap \"Not them\" and Earshield learns that voice is fine.")
        status.recentBlocks.take(8).forEach { e ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(e.name)
                    Hint(
                        fmt.format(Date(e.timeMillis)) + " · ${(e.score * 100).roundToInt()}% match" +
                            if (e.leakedSeconds > 0.05f) " · first ${"%.1f".format(e.leakedSeconds)} s got through" else ""
                    )
                }
                TextButton(onClick = { onNotThem(e.id) }, enabled = !e.corrected) {
                    Text(if (e.corrected) "Marked" else "Not them")
                }
            }
        }
    }
}

@Composable
private fun AddVoiceDialog(
    controller: ShieldController,
    status: LiveStatus,
    role: VoiceRole,
    onStartListening: () -> Unit,
    onDismiss: () -> Unit,
) {
    val allowed = role != VoiceRole.BLOCK
    val isMe = role == VoiceRole.ME
    var name by remember { mutableStateOf(if (isMe) "You" else "") }
    var error by remember { mutableStateOf<String?>(null) }
    var mixed by remember { mutableStateOf(false) }
    var saving by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val recording = status.enrollingSeconds
    val target = 12f
    val whose = if (isMe) "your" else "their"

    fun close() {
        controller.cancelEnrollment()
        onDismiss()
    }

    fun handle(outcome: SaveOutcome) {
        saving = false
        when (outcome) {
            SaveOutcome.Saved -> onDismiss()
            SaveOutcome.Mixed -> mixed = true
            is SaveOutcome.Failed -> error = outcome.message
        }
    }

    AlertDialog(
        onDismissRequest = ::close,
        title = { Text(when (role) { VoiceRole.ME -> "Teach it your voice"; VoiceRole.ALLOW -> "Allow a voice"; else -> "Add a voice to block" }) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("Name") }, singleLine = true,
                    modifier = Modifier.fillMaxWidth())
                when {
                    mixed -> {
                        Text("This recording sounds like more than one voice, or a lot of background noise. " +
                            "A voiceprint like that can end up matching everyone.",
                            style = MaterialTheme.typography.bodyMedium)
                        Hint("Record again while only ${if (isMe) "you are" else "they are"} talking, or save it anyway.")
                    }
                    !status.running -> {
                        Hint("Earshield needs to be listening to learn a voice.")
                        Button(onClick = onStartListening, enabled = !status.starting) { Text("Start listening") }
                    }
                    recording != null -> {
                        MicMeter(status)
                        Text("Recording $whose voice: ${"%.0f".format(recording)} s of speech",
                            style = MaterialTheme.typography.bodyMedium)
                        LinearProgressIndicator(progress = { (recording / target).coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                        Hint(
                            if (recording < Enrollment.MIN_SECONDS) {
                                if (isMe) "Keep talking normally. Read something aloud if it helps."
                                else "Keep going while only they are talking."
                            } else "That's enough to work. Around ${target.toInt()} s is more reliable."
                        )
                    }
                    else -> {
                        MicMeter(status)
                        Hint(
                            if (isMe) "Talk normally for about 12 seconds with no one else talking. " +
                                "Earshield will never cover your voice."
                            else "Record while only this person is talking. Other voices mixed in make blocking less accurate. " +
                                "The bar turns pink when Earshield hears speech."
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = { error = null; controller.startEnrollment() }) { Text("Record now") }
                            if (!isMe) {
                                OutlinedButton(onClick = {
                                    saving = true
                                    error = null
                                    scope.launch { handle(controller.enrollFromRecent(name, allowed)) }
                                }, enabled = !saving) { Text("Use last 10 s") }
                            }
                        }
                        if (!isMe) {
                            Hint("\"Use last 10 s\" learns whoever spoke most recently. Only use it if they were the only one talking.")
                        }
                    }
                }
                error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
            }
        },
        confirmButton = {
            when {
                mixed -> TextButton(onClick = { controller.saveAnyway(name, allowed); onDismiss() }) { Text("Save anyway") }
                recording != null -> TextButton(
                    enabled = recording >= Enrollment.MIN_SECONDS && !saving,
                    onClick = {
                        saving = true
                        scope.launch { handle(controller.finishEnrollment(name, allowed)) }
                    },
                ) { Text("Save") }
            }
        },
        dismissButton = {
            if (mixed) {
                TextButton(onClick = { mixed = false; error = null; controller.startEnrollment() }) { Text("Record again") }
            } else {
                TextButton(onClick = ::close) { Text("Cancel") }
            }
        },
    )
}

/** Record a conversation, find the people in it, listen to each, and choose block / allow / me / skip. */
@Composable
private fun LearnVoicesDialog(
    controller: ShieldController,
    status: LiveStatus,
    onStartListening: () -> Unit,
    onDismiss: () -> Unit,
) {
    val scope = rememberCoroutineScope()
    var results by remember { mutableStateOf<List<DiscoveredVoice>?>(null) }
    val names = remember { mutableStateOf(mapOf<Int, String>()) }
    val roles = remember { mutableStateOf(mapOf<Int, VoiceRole>()) }
    val recording = status.captureSeconds
    val analyzing = status.discoveryProgress

    fun close() {
        controller.cancelCapture()
        onDismiss()
    }

    AlertDialog(
        onDismissRequest = {},
        title = { Text("Learn voices from a conversation") },
        text = {
            Column(
                Modifier.verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                val found = results
                when {
                    !status.running -> {
                        Hint("Earshield needs to be listening to record.")
                        Button(onClick = onStartListening, enabled = !status.starting) { Text("Start listening") }
                    }
                    analyzing != null -> {
                        Text("Finding the voices…", style = MaterialTheme.typography.bodyMedium)
                        LinearProgressIndicator(progress = { analyzing }, modifier = Modifier.fillMaxWidth())
                    }
                    found != null && found.isEmpty() -> {
                        Text("No clear voices were found. Try recording a little longer, closer to the people talking.")
                    }
                    found != null -> {
                        Hint("Listen to each voice and choose what to do. The recording is deleted when you close this.")
                        found.forEach { v -> FoundVoiceCard(v, controller, names, roles) }
                    }
                    recording != null -> {
                        MicMeter(status)
                        val max = VoiceDiscovery.MAX_RECORDING_SECONDS
                        Text("Recording: ${"%.0f".format(recording)} s", style = MaterialTheme.typography.bodyMedium)
                        LinearProgressIndicator(progress = { recording / max }, modifier = Modifier.fillMaxWidth())
                        Hint(
                            if (recording < VoiceDiscovery.MIN_RECORDING_SECONDS) "Keep going: at least ${VoiceDiscovery.MIN_RECORDING_SECONDS.toInt()} s."
                            else if (recording >= max) "That's the most it can record. Tap \"Find voices\"."
                            else "Each person should say a few sentences. Up to ${max.toInt()} s."
                        )
                    }
                    else -> {
                        MicMeter(status)
                        Hint("Record 30–90 s of normal conversation. People can talk over each other. Each person " +
                            "should say a few sentences. Nothing is saved except the voiceprints you choose to keep.")
                        Button(onClick = { controller.startCapture() }) { Text("Start recording") }
                    }
                }
            }
        },
        confirmButton = {
            val found = results
            when {
                recording != null -> TextButton(
                    enabled = recording >= VoiceDiscovery.MIN_RECORDING_SECONDS,
                    onClick = { scope.launch { results = controller.findVoices() } },
                ) { Text("Find voices") }
                found != null && found.isNotEmpty() -> TextButton(onClick = {
                    controller.saveDiscovered(found.map { v ->
                        Triple(v, names.value[v.number] ?: "", roles.value[v.number] ?: VoiceRole.SKIP)
                    })
                    onDismiss()
                }) { Text("Save") }
            }
        },
        dismissButton = { TextButton(onClick = ::close, enabled = analyzing == null) { Text(if (results != null) "Close" else "Cancel") } },
    )
}

@Composable
private fun FoundVoiceCard(
    v: DiscoveredVoice,
    controller: ShieldController,
    names: androidx.compose.runtime.MutableState<Map<Int, String>>,
    roles: androidx.compose.runtime.MutableState<Map<Int, VoiceRole>>,
) {
    val role = roles.value[v.number] ?: VoiceRole.SKIP
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerHigh)) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("Voice ${v.number}", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                    Hint("${"%.0f".format(v.found.speechSeconds)} s of speech found")
                }
                FilledTonalButton(onClick = { controller.playClip(v.found.clip) }) {
                    Icon(Icons.Filled.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("Play")
                }
            }
            v.knownAs?.let { Hint("Sounds like ${it.name} (${(v.knownScore * 100).toInt()}% match), who is already saved.") }
            v.maybeSameAs?.let { Hint("Might be the same person as Voice $it. If so, give both the same name and choice.") }
            if (v.found.consistency < 0.74f) Hint("This one may be a mix of people. Listen before choosing.")
            val choices = listOf(VoiceRole.BLOCK to "Block", VoiceRole.ALLOW to "Allow", VoiceRole.ME to "Me", VoiceRole.SKIP to "Skip")
            SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                choices.forEachIndexed { i, (r, label) ->
                    SegmentedButton(
                        selected = role == r,
                        onClick = { roles.value = roles.value + (v.number to r) },
                        shape = SegmentedButtonDefaults.itemShape(i, choices.size),
                    ) { Text(label) }
                }
            }
            if (role == VoiceRole.BLOCK || role == VoiceRole.ALLOW) {
                OutlinedTextField(
                    value = names.value[v.number] ?: "",
                    onValueChange = { names.value = names.value + (v.number to it) },
                    label = { Text("Name") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
    }
}

/** Live speaker identification: who the app is hearing, and who they are. */
@Composable
private fun WhoIsTalkingCard(status: LiveStatus, controller: ShieldController) {
    var saving by remember { mutableStateOf<LiveRow?>(null) }
    SectionCard {
        SectionTitle("Who's talking")
        if (status.liveSpeakers.isEmpty()) {
            Hint("No one yet. People appear here a second or two after they start talking.")
        }
        status.liveSpeakers.forEach { row ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                val dot = when {
                    row.covered && row.talkingNow -> TransPink
                    row.talkingNow -> TransBlue
                    else -> MaterialTheme.colorScheme.outline
                }
                Box(Modifier.size(10.dp).background(dot, CircleShape))
                Spacer(Modifier.width(10.dp))
                Column(Modifier.weight(1f)) {
                    Text(row.label, style = MaterialTheme.typography.bodyLarge,
                        fontWeight = if (row.talkingNow) FontWeight.SemiBold else FontWeight.Normal)
                    Hint(buildString {
                        append(when (row.role) {
                            LiveSpeaker.Role.BLOCKED -> "Blocked"
                            LiveSpeaker.Role.ALLOWED -> "Allowed"
                            LiveSpeaker.Role.UNKNOWN -> "Not saved"
                            LiveSpeaker.Role.BACKGROUND -> "Background talker"
                        })
                        if (row.role != LiveSpeaker.Role.UNKNOWN) append(" · ${(row.score * 100).toInt()}% match")
                        if (row.talkingNow) append(if (row.covered) " · talking, covered" else " · talking")
                    })
                }
                if (row.trackId != null) {
                    TextButton(onClick = { saving = row }) { Text("Save") }
                }
            }
        }
        Hint("Live audio is analyzed before you hear it, so names can appear a moment before the voice does.")
    }
    saving?.let { row -> SaveLiveVoiceDialog(row, controller) { saving = null } }
}

@Composable
private fun SaveLiveVoiceDialog(row: LiveRow, controller: ShieldController, onDismiss: () -> Unit) {
    var name by remember { mutableStateOf("") }
    var allowed by remember { mutableStateOf(false) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Save ${row.label}") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Hint("Uses everything this person has said so far. Save once they've talked for a few seconds.")
                OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("Name") },
                    singleLine = true, modifier = Modifier.fillMaxWidth())
                SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                    SegmentedButton(selected = !allowed, onClick = { allowed = false },
                        shape = SegmentedButtonDefaults.itemShape(0, 2)) { Text("Block") }
                    SegmentedButton(selected = allowed, onClick = { allowed = true },
                        shape = SegmentedButtonDefaults.itemShape(1, 2)) { Text("Allow") }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                row.trackId?.let { controller.saveLiveVoice(it, name, allowed) }
                onDismiss()
            }) { Text("Save") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}
