package app.earshield.app.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val TransBlue = Color(0xFF5BCEFA)
val TransPink = Color(0xFFF5A9B8)

private val Dark = darkColorScheme(
    primary = TransBlue,
    onPrimary = Color(0xFF00344A),
    primaryContainer = Color(0xFF004C6A),
    onPrimaryContainer = Color(0xFFC6E7FF),
    secondary = TransPink,
    onSecondary = Color(0xFF4A1824),
    secondaryContainer = Color(0xFF6B2E3B),
    onSecondaryContainer = Color(0xFFFFD9DF),
    background = Color(0xFF1B1530),
    onBackground = Color(0xFFE8E0F0),
    surface = Color(0xFF1B1530),
    onSurface = Color(0xFFE8E0F0),
    surfaceContainer = Color(0xFF261F3D),
    surfaceContainerHigh = Color(0xFF302848),
    onSurfaceVariant = Color(0xFFC9C0D6),
)

private val Light = lightColorScheme(
    primary = Color(0xFF00658C),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFC6E7FF),
    onPrimaryContainer = Color(0xFF001E2D),
    secondary = Color(0xFF9A4558),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFFFD9DF),
    onSecondaryContainer = Color(0xFF3F0217),
    background = Color(0xFFFCF8FF),
    surface = Color(0xFFFCF8FF),
    surfaceContainer = Color(0xFFF1ECF6),
    surfaceContainerHigh = Color(0xFFEBE6F0),
)

@Composable
fun EarshieldTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = if (isSystemInDarkTheme()) Dark else Light, content = content)
}
