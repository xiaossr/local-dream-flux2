package io.github.xororz.localdream.service

import android.app.*
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import io.github.xororz.localdream.R
import io.github.xororz.localdream.data.Model
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File
import java.util.concurrent.TimeUnit
import io.github.xororz.localdream.data.ModelRepository

class BackendService : Service() {
    private var process: Process? = null

    companion object {
        private const val TAG = "BackendService"
        private const val EXECUTABLE_NAME = "libstable_diffusion_core.so"
        private const val NOTIFICATION_ID = 2
        private const val CHANNEL_ID = "backend_service_channel"

        const val ACTION_STOP = "io.github.xororz.localdream.STOP_GENERATION"
        const val ACTION_RESTART = "io.github.xororz.localdream.RESTART_BACKEND"

        private object StateHolder {
            val _backendState = MutableStateFlow<BackendState>(BackendState.Idle)
        }

        val backendState: StateFlow<BackendState> = StateHolder._backendState

        private fun updateState(state: BackendState) {
            StateHolder._backendState.value = state
        }
    }

    sealed class BackendState {
        object Idle : BackendState()
        object Starting : BackendState()
        object Running : BackendState()
        data class Error(val message: String) : BackendState()
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onBind(intent: Intent): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(TAG, "service started command: ${intent?.action}")
        startForeground(
            NOTIFICATION_ID,
            createNotification(this.getString(R.string.backend_notify))
        )

        when (intent?.action) {
            ACTION_STOP -> {
                Log.d("GenerationService", "stop")
                stopSelf()
                return START_NOT_STICKY
            }

            ACTION_RESTART -> {
                Log.i(TAG, "restarting backend service")
                stopBackend()
            }
        }

        val modelId = intent?.getStringExtra("modelId")
        val width = intent?.getIntExtra("width", 512) ?: 512
        val height = intent?.getIntExtra("height", 512) ?: 512
        if (modelId != null) {
            val modelRepository = ModelRepository(this)
            val model = modelRepository.models.find { it.id == modelId }

            if (model != null) {
                if (startBackend(model, width, height)) {
                    updateState(BackendState.Running)
                } else {
                    updateState(BackendState.Error("Backend start failed"))
                }
            } else {
                updateState(BackendState.Error("Model not found"))
                stopSelf()
            }
        }

        return START_NOT_STICKY
    }

    override fun onTimeout(startId: Int) {
        super.onTimeout(startId)
        Log.e(TAG, "Foreground service timeout")
        updateState(BackendState.Error("Service timeout"))
        stopBackend()
        stopSelf()
    }

    private fun createNotificationChannel() {
        val name = "Backend Service"
        val descriptionText = "Backend service for image generation"
        val importance = NotificationManager.IMPORTANCE_LOW
        val channel = NotificationChannel(CHANNEL_ID, name, importance).apply {
            description = descriptionText
        }
        val notificationManager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.createNotificationChannel(channel)
    }

    private fun createNotification(contentText: String): Notification {
        val openAppIntent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
        }
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            openAppIntent,
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(this.getString(R.string.backend_notify_title))
            .setContentText(contentText)
            .setSmallIcon(R.drawable.ic_launcher_monochrome)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun startBackend(model: Model, width: Int, height: Int): Boolean {
        Log.i(TAG, "backend start, model: ${model.name}, resolution: ${width}×${height}")
        updateState(BackendState.Starting)

        try {
            val nativeDir = applicationInfo.nativeLibraryDir
            val modelsDir = File(Model.getModelsDir(this), model.id)

            val executableFile = File(nativeDir, EXECUTABLE_NAME)

            if (!executableFile.exists()) {
                Log.e(TAG, "error: executable does not exist: ${executableFile.absolutePath}")
                return false
            }

            // ExecuTorch FLUX.2 pipeline: 3 .pte models + tokenizer + config
            var command = listOf(
                executableFile.absolutePath,
                "--encoder", File(modelsDir, "text_encoder.pte").absolutePath,
                "--dit", File(modelsDir, "transformer.pte").absolutePath,
                "--vae_decoder", File(modelsDir, "vae_decoder.pte").absolutePath,
                "--tokenizer", File(modelsDir, "tokenizer/tokenizer.json").absolutePath,
                "--config", File(modelsDir, "export_config.json").absolutePath,
                "--bn_stats", File(modelsDir, "vae_bn_stats.json").absolutePath,
                "--port", "8081"
            )

            // Optional img2img models
            val vaeEncoderFile = File(modelsDir, "vae_encoder.pte")
            val ditImg2imgFile = File(modelsDir, "transformer_img2img.pte")
            if (vaeEncoderFile.exists() && ditImg2imgFile.exists()) {
                command = command + listOf(
                    "--vae_encoder", vaeEncoderFile.absolutePath,
                    "--dit_img2img", ditImg2imgFile.absolutePath
                )
                Log.i(TAG, "img2img models found, enabling img2img support")
            }

            val env = mutableMapOf<String, String>()
            env["LD_LIBRARY_PATH"] = listOf(
                nativeDir,
                "/system/lib64",
                "/vendor/lib64",
            ).joinToString(":")

            Log.d(TAG, "COMMAND: ${command.joinToString(" ")}")
            Log.d(TAG, "LD_LIBRARY_PATH=${env["LD_LIBRARY_PATH"]}")

            val processBuilder = ProcessBuilder(command).apply {
                directory(File(nativeDir))
                redirectErrorStream(true)
                environment().putAll(env)
            }

            process = processBuilder.start()

            startMonitorThread()

            return true

        } catch (e: Exception) {
            Log.e(TAG, "backend start failed", e)
            updateState(BackendState.Error("backend start failed: ${e.message}"))
            e.printStackTrace()
            return false
        }
    }

    private fun startMonitorThread() {
        Thread {
            try {
                process?.let { proc ->
                    proc.inputStream.bufferedReader().use { reader ->
                        var line: String?
                        while (reader.readLine().also { line = it } != null) {
                            Log.i(TAG, "Backend: $line")
                        }
                    }

                    val exitCode = proc.waitFor()
                    Log.i(TAG, "Backend process exited with code: $exitCode")
                    updateState(BackendState.Error("Backend process exited with code: $exitCode"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "monitor error", e)
                updateState(BackendState.Error("monitor error: ${e.message}"))
            }
        }.apply {
            isDaemon = true
            start()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopBackend()
    }

    fun stopBackend() {
        Log.i(TAG, "to stop backend")
        process?.let { proc ->
            try {
                proc.destroy()

                if (!proc.waitFor(5, TimeUnit.SECONDS)) {
                    proc.destroyForcibly()
                }

                Log.i(TAG, "process end, code: ${proc.exitValue()}")
                updateState(BackendState.Idle)
            } catch (e: Exception) {
                Log.e(TAG, "error", e)
                updateState(BackendState.Error("error: ${e.message}"))
            } finally {
                process = null
            }
        }
    }
}