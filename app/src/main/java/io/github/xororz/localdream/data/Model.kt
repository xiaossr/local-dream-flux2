package io.github.xororz.localdream.data

import android.content.Context
import android.os.Build
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import io.github.xororz.localdream.R
import io.github.xororz.localdream.service.ModelDownloadService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.io.File
import android.content.Intent
import android.util.Log

data class Resolution(
    val width: Int,
    val height: Int
) {
    val isSquare: Boolean get() = width == height

    override fun toString(): String =
        if (isSquare) "${width}×${width}"
        else "${width}×${height}"
}

object PatchScanner {
    private val squarePatchPattern = Regex("""^(\d+)\.patch$""")
    private val rectangularPatchPattern = Regex("""^(\d+)x(\d+)\.patch$""")

    fun scanAvailableResolutions(context: Context, modelId: String): List<Resolution> {
        val modelDir = File(Model.getModelsDir(context), modelId)
        if (!modelDir.exists() || !modelDir.isDirectory) {
            return emptyList()
        }

        val resolutions = mutableListOf<Resolution>()

        modelDir.listFiles()?.forEach { file ->
            if (!file.isFile) return@forEach

            squarePatchPattern.matchEntire(file.name)?.let { match ->
                val size = match.groupValues[1].toIntOrNull()
                if (size != null && size > 0) {
                    resolutions.add(Resolution(size, size))
                }
            }

            rectangularPatchPattern.matchEntire(file.name)?.let { match ->
                val width = match.groupValues[1].toIntOrNull()
                val height = match.groupValues[2].toIntOrNull()
                if (width != null && height != null && width > 0 && height > 0) {
                    resolutions.add(Resolution(width, height))
                }
            }
        }

        return resolutions.sortedBy { it.width * it.height }.distinct()
    }
}

private fun getDeviceSoc(): String {
    return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        Build.SOC_MODEL
    } else {
        "CPU"
    }
}

data class DownloadProgress(
    val progress: Float,
    val downloadedBytes: Long,
    val totalBytes: Long
)

val chipsetModelSuffixes = mapOf(
    "SM8475" to "8gen1",
    "SM8450" to "8gen1",
    "SM8550" to "8gen2",
    "SM8550P" to "8gen2",
    "QCS8550" to "8gen2",
    "QCM8550" to "8gen2",
    "SM8650" to "8gen2",
    "SM8650P" to "8gen2",
    "SM8750" to "8gen2",
    "SM8750P" to "8gen2",
    "SM8850" to "8gen2",
    "SM8850P" to "8gen2",
    "SM8735" to "8gen2",
    "SM8845" to "8gen2",
)

sealed class DownloadResult {
    data object Success : DownloadResult()
    data class Error(val message: String) : DownloadResult()
    data class Progress(val progress: DownloadProgress) : DownloadResult()
}

data class Model(
    val id: String,
    val name: String,
    val description: String,
    val baseUrl: String,
    val fileUri: String = "",
    val generationSize: Int = 512,
    val approximateSize: String = "1GB",
    val isDownloaded: Boolean = false,
    val defaultPrompt: String = "",
    val isCustom: Boolean = false,
    val runOnCpu: Boolean = true

) {

    fun startDownload(context: Context) {
        if (isCustom || fileUri.isEmpty()) return

        val intent = Intent(context, ModelDownloadService::class.java).apply {
            action = ModelDownloadService.ACTION_START_DOWNLOAD
            putExtra(ModelDownloadService.EXTRA_MODEL_ID, id)
            putExtra(ModelDownloadService.EXTRA_MODEL_NAME, name)
            putExtra(ModelDownloadService.EXTRA_FILE_URL, "${baseUrl.removeSuffix("/")}/$fileUri")
            putExtra(ModelDownloadService.EXTRA_IS_ZIP, fileUri.endsWith(".zip"))
            putExtra(ModelDownloadService.EXTRA_MODEL_TYPE, "flux2")
        }

        context.startForegroundService(intent)
    }

    fun deleteModel(context: Context): Boolean {
        return try {
            val modelDir = File(getModelsDir(context), id)
            val historyManager = HistoryManager(context)
            val generationPreferences = GenerationPreferences(context)

            runBlocking {
                historyManager.clearHistoryForModel(id)
                generationPreferences.clearPreferencesForModel(id)
            }

            if (modelDir.exists() && modelDir.isDirectory) {
                val deleted = modelDir.deleteRecursively()
                Log.d("Model", "Delete model $id: $deleted")
                deleted
            } else {
                Log.d("Model", "Model does not exist: $id")
                false
            }
        } catch (e: Exception) {
            Log.e("Model", "error: ${e.message}")
            false
        }
    }

    companion object {
        private const val MODELS_DIR = "models"

        fun isDeviceSupported(): Boolean {
            val soc = getDeviceSoc()
            return getChipsetSuffix(soc) != null
        }

        fun isQualcommDevice(): Boolean {
            val soc = getDeviceSoc()
            return soc.startsWith("SM") || soc.startsWith("QCS") || soc.startsWith("QCM")
        }

        fun getChipsetSuffix(soc: String): String? {
            if (soc in chipsetModelSuffixes) {
                return chipsetModelSuffixes[soc]
            }
            if (soc.startsWith("SM")) {
                return "min"
            }
            return null
        }

        fun getModelsDir(context: Context): File {
            // Use /data/local/tmp for fast I/O — accessible via adb push without root
            return File("/data/local/tmp/localdream", MODELS_DIR).apply {
                if (!exists()) mkdirs()
            }
        }

        fun isModelDownloaded(
            context: Context,
            modelId: String,
            isCustom: Boolean = false
        ): Boolean {
            if (isCustom) {
                return true
            }

            val modelDir = File(getModelsDir(context), modelId)
            if (!modelDir.exists() || !modelDir.isDirectory) {
                return false
            }

            // Check for FLUX.2 model marker
            val configFile = File(modelDir, "export_config.json")
            return configFile.exists()
        }
    }
}

data class UpscalerModel(
    val id: String,
    val name: String,
    val description: String,
    val baseUrl: String,
    val fileUri: String,
    val isDownloaded: Boolean = false
) {
    fun startDownload(context: Context) {
        val intent = Intent(context, ModelDownloadService::class.java).apply {
            action = ModelDownloadService.ACTION_START_DOWNLOAD
            putExtra(ModelDownloadService.EXTRA_MODEL_ID, id)
            putExtra(ModelDownloadService.EXTRA_MODEL_NAME, name)
            putExtra(ModelDownloadService.EXTRA_FILE_URL, "${baseUrl.removeSuffix("/")}/$fileUri")
            putExtra(ModelDownloadService.EXTRA_IS_ZIP, false)
            putExtra(ModelDownloadService.EXTRA_MODEL_TYPE, "upscaler")
        }

        context.startForegroundService(intent)
    }
}

class UpscalerRepository(private val context: Context) {
    private val generationPreferences = GenerationPreferences(context)

    private var _baseUrl = mutableStateOf("https://huggingface.co/")
    var baseUrl: String
        get() = _baseUrl.value
        private set(value) {
            _baseUrl.value = value
        }

    var upscalers by mutableStateOf(initializeUpscalers())
        private set

    init {
        CoroutineScope(Dispatchers.Main).launch {
            baseUrl = generationPreferences.getBaseUrl()
            upscalers = initializeUpscalers()
        }
    }

    private fun initializeUpscalers(): List<UpscalerModel> {
        val soc = getDeviceSoc()
        val suffix = Model.getChipsetSuffix(soc) ?: "min"

        return listOf(
            createAnimeUpscaler(suffix),
            createRealisticUpscaler(suffix)
        )
    }

    private fun createAnimeUpscaler(suffix: String): UpscalerModel {
        val id = "upscaler_anime"
        val fileUri =
            "xororz/upscaler/resolve/main/realesrgan_x4plus_anime_6b/upscaler_${suffix}.bin"

        val isDownloaded = Model.isModelDownloaded(context, id, false)

        return UpscalerModel(
            id = id,
            name = context.getString(R.string.upscaler_anime),
            description = context.getString(R.string.upscaler_anime_desc),
            baseUrl = baseUrl,
            fileUri = fileUri,
            isDownloaded = isDownloaded
        )
    }

    private fun createRealisticUpscaler(suffix: String): UpscalerModel {
        val id = "upscaler_realistic"
        val fileUri = "xororz/upscaler/resolve/main/4x_UltraSharpV2_Lite/upscaler_${suffix}.bin"

        val isDownloaded = Model.isModelDownloaded(context, id, false)

        return UpscalerModel(
            id = id,
            name = context.getString(R.string.upscaler_realistic),
            description = context.getString(R.string.upscaler_realistic_desc),
            baseUrl = baseUrl,
            fileUri = fileUri,
            isDownloaded = isDownloaded
        )
    }

    fun refreshUpscalerState(upscalerId: String) {
        upscalers = upscalers.map { upscaler ->
            if (upscaler.id == upscalerId) {
                val isDownloaded = Model.isModelDownloaded(context, upscaler.id, false)
                upscaler.copy(isDownloaded = isDownloaded)
            } else {
                upscaler
            }
        }
    }
}

class ModelRepository(private val context: Context) {
    private val generationPreferences = GenerationPreferences(context)

    private var _baseUrl = mutableStateOf("https://huggingface.co/")
    var baseUrl: String
        get() = _baseUrl.value
        private set(value) {
            _baseUrl.value = value
        }

    var models by mutableStateOf(initializeModels())
        private set

    init {
        CoroutineScope(Dispatchers.Main).launch {
            baseUrl = generationPreferences.getBaseUrl()
            models = initializeModels()
        }
    }

    private fun scanCustomModels(): List<Model> {
        val modelsDir = Model.getModelsDir(context)
        val customModels = mutableListOf<Model>()

        if (modelsDir.exists() && modelsDir.isDirectory) {
            modelsDir.listFiles()?.forEach { dir ->
                if (dir.isDirectory) {
                    // Check for FLUX.2 model marker (export_config.json)
                    val configFile = File(dir, "export_config.json")
                    if (configFile.exists()) {
                        val customModel = createCustomModel(dir)
                        customModels.add(customModel)
                    }
                }
            }
        }

        return customModels.sortedBy { it.name.lowercase() }
    }

    private fun createCustomModel(modelDir: File): Model {
        val modelId = modelDir.name

        return Model(
            id = modelId,
            name = modelId,
            description = context.getString(R.string.custom_model),
            baseUrl = "",
            approximateSize = "Custom",
            isDownloaded = true,
            defaultPrompt = "a beautiful landscape, high quality",
            isCustom = true
        )
    }

    private fun initializeModels(): List<Model> {
        val customModels = scanCustomModels()
        val customIds = customModels.map { it.id }.toSet()

        val predefinedModels = mutableListOf(
            createFlux2KleinModel(),
        ).filter { it.id !in customIds }

        return customModels + predefinedModels
    }

    private fun createFlux2KleinModel(): Model {
        val id = "flux2klein"
        val fileUri = "xororz/flux2-klein-executorch/resolve/main/flux2_klein_int8.zip"

        val isDownloaded = Model.isModelDownloaded(context, id, false)

        return Model(
            id = id,
            name = "FLUX.2-klein",
            description = context.getString(R.string.flux2klein_description),
            baseUrl = baseUrl,
            fileUri = fileUri,
            generationSize = 512,
            approximateSize = "4.5GB",
            isDownloaded = isDownloaded,
            defaultPrompt = "a beautiful landscape with mountains and a lake, high quality, detailed"
        )
    }

    fun refreshModelState(modelId: String) {
        models = models.map { model ->
            if (model.id == modelId) {
                val isDownloaded = Model.isModelDownloaded(context, modelId, model.isCustom)
                model.copy(isDownloaded = isDownloaded)
            } else {
                model
            }
        }
    }

    fun refreshAllModels() {
        models = initializeModels()
    }
}