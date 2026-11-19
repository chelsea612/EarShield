plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "app.earshield"
    compileSdk = 36

    defaultConfig {
        applicationId = "app.earshield"
        minSdk = 26
        targetSdk = 36
        versionCode = 12
        versionName = "0.7.0"
        // Phones only; leaving out x86 emulator libraries keeps the APK much smaller.
        // 64-bit phones only: keeps the APK small. (Practically every phone since about 2017.)
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    buildTypes {
        release {
            // Shrink the app's code. ONNX Runtime is kept whole: its native code calls into it by name.
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // Signed with the debug key so the release build can be sideloaded for testing.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures { compose = true }

    // The voice encoder and VAD models live in ../models and are bundled as assets.
    sourceSets["main"].assets.srcDir(rootProject.file("models"))
    // Compress the models and native libraries inside the APK to keep the download small.
    // (Models are read fully into memory anyway; libraries are unpacked once at install.)
    packaging { jniLibs { useLegacyPackaging = true } }
}

kotlin {
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
}

dependencies {
    implementation(project(":core"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.activity:activity-compose:1.11.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.4")
    implementation(platform("androidx.compose:compose-bom:2025.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
}
