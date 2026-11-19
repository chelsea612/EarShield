// Platform-independent audio logic: features, voice matching, and the blocking pipeline.
// Runs on the JVM so it can be tested on a computer with real recordings.
plugins {
    id("org.jetbrains.kotlin.jvm")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}
kotlin {
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
}

val onnxRuntimeVersion = "1.22.0"

dependencies {
    // The Android app supplies onnxruntime-android, which has the same Java API.
    compileOnly("com.microsoft.onnxruntime:onnxruntime:$onnxRuntimeVersion")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:$onnxRuntimeVersion")
    testImplementation("org.json:json:20240303")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
    systemProperty("models.dir", rootProject.file("models").absolutePath)
    System.getProperty("earshield.debug")?.let { systemProperty("earshield.debug", it) }
    testLogging { showStandardStreams = true; events("passed", "failed") }
}
