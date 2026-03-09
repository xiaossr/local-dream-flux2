package io.github.xororz.localdream.ui.screens

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect as AndroidRect
import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.exifinterface.media.ExifInterface
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import io.github.xororz.localdream.R
import java.util.Base64

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CropImageScreen(
    imageUri: Uri,
    width: Int,
    height: Int,
    onCropComplete: (String, Bitmap, AndroidRect) -> Unit,
    onCancel: () -> Unit
) {
    val context = LocalContext.current
    var sourceBitmap by remember { mutableStateOf<Bitmap?>(null) }

    // Crop rect position in canvas coordinates
    var cropLeft by remember { mutableFloatStateOf(0f) }
    var cropTop by remember { mutableFloatStateOf(0f) }
    var cropSize by remember { mutableFloatStateOf(0f) }

    // Image draw info (computed on layout)
    var imgDrawLeft by remember { mutableFloatStateOf(0f) }
    var imgDrawTop by remember { mutableFloatStateOf(0f) }
    var imgDrawWidth by remember { mutableFloatStateOf(0f) }
    var imgDrawHeight by remember { mutableFloatStateOf(0f) }

    val targetAspect = width.toFloat() / height.toFloat()

    LaunchedEffect(imageUri) {
        try {
            val rawBitmap = context.contentResolver.openInputStream(imageUri)?.use {
                BitmapFactory.decodeStream(it)
            }
            val rotation = context.contentResolver.openInputStream(imageUri)?.use {
                val exif = ExifInterface(it)
                when (exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                    ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                    ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                    ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> 0f
                    ExifInterface.ORIENTATION_FLIP_VERTICAL -> 180f
                    else -> 0f
                }
            } ?: 0f
            sourceBitmap = if (rawBitmap != null && rotation != 0f) {
                val matrix = Matrix().apply { postRotate(rotation) }
                Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
            } else {
                rawBitmap
            }
        } catch (_: Exception) {
            sourceBitmap = null
        }
    }

    BackHandler { onCancel() }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.crop_image)) },
                navigationIcon = {
                    IconButton(onClick = onCancel) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                },
                actions = {
                    if (sourceBitmap != null) {
                        IconButton(onClick = {
                            val bmp = sourceBitmap ?: return@IconButton
                            val scaleX = bmp.width.toFloat() / imgDrawWidth
                            val scaleY = bmp.height.toFloat() / imgDrawHeight

                            val cropW = if (targetAspect >= 1f) cropSize else cropSize * targetAspect
                            val cropH = if (targetAspect >= 1f) cropSize / targetAspect else cropSize

                            val srcLeft = ((cropLeft - imgDrawLeft) * scaleX).toInt().coerceIn(0, bmp.width - 1)
                            val srcTop = ((cropTop - imgDrawTop) * scaleY).toInt().coerceIn(0, bmp.height - 1)
                            val srcW = (cropW * scaleX).toInt().coerceIn(1, bmp.width - srcLeft)
                            val srcH = (cropH * scaleY).toInt().coerceIn(1, bmp.height - srcTop)

                            val cropped = Bitmap.createBitmap(bmp, srcLeft, srcTop, srcW, srcH)
                            val scaled = Bitmap.createScaledBitmap(cropped, width, height, true)
                            if (cropped !== scaled) cropped.recycle()

                            val rect = AndroidRect(srcLeft, srcTop, srcLeft + srcW, srcTop + srcH)
                            val base64 = bitmapToRawRgbBase64(scaled, width, height)

                            onCropComplete(base64, scaled, rect)
                        }) {
                            Icon(
                                imageVector = Icons.Default.Check,
                                contentDescription = "Confirm crop"
                            )
                        }
                    }
                }
            )
        }
    ) { paddingValues ->
        val bmp = sourceBitmap
        if (bmp == null) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "Loading image...",
                    style = MaterialTheme.typography.bodyLarge
                )
            }
        } else {
            val imageBitmap = remember(bmp) { bmp.asImageBitmap() }

            Canvas(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .pointerInput(bmp, targetAspect) {
                        detectDragGestures { change, dragAmount ->
                            change.consume()
                            val cropW =
                                if (targetAspect >= 1f) cropSize else cropSize * targetAspect
                            val cropH =
                                if (targetAspect >= 1f) cropSize / targetAspect else cropSize

                            cropLeft = (cropLeft + dragAmount.x)
                                .coerceIn(imgDrawLeft, imgDrawLeft + imgDrawWidth - cropW)
                            cropTop = (cropTop + dragAmount.y)
                                .coerceIn(imgDrawTop, imgDrawTop + imgDrawHeight - cropH)
                        }
                    }
            ) {
                val canvasW = size.width
                val canvasH = size.height
                val bmpAspect = bmp.width.toFloat() / bmp.height.toFloat()

                val drawW: Float
                val drawH: Float
                if (bmpAspect > canvasW / canvasH) {
                    drawW = canvasW
                    drawH = canvasW / bmpAspect
                } else {
                    drawH = canvasH
                    drawW = canvasH * bmpAspect
                }
                val drawLeft = (canvasW - drawW) / 2f
                val drawTop = (canvasH - drawH) / 2f

                imgDrawLeft = drawLeft
                imgDrawTop = drawTop
                imgDrawWidth = drawW
                imgDrawHeight = drawH

                drawImage(
                    image = imageBitmap,
                    srcOffset = IntOffset.Zero,
                    srcSize = IntSize(bmp.width, bmp.height),
                    dstOffset = IntOffset(drawLeft.toInt(), drawTop.toInt()),
                    dstSize = IntSize(drawW.toInt(), drawH.toInt())
                )

                // Initialize crop rect to fit-centered within the image area
                if (cropSize == 0f) {
                    val maxCropW: Float
                    val maxCropH: Float
                    if (targetAspect >= 1f) {
                        maxCropW = drawW
                        maxCropH = drawW / targetAspect
                        if (maxCropH > drawH) {
                            val s = drawH * targetAspect
                            cropSize = s
                        } else {
                            cropSize = drawW
                        }
                    } else {
                        maxCropH = drawH
                        maxCropW = drawH * targetAspect
                        if (maxCropW > drawW) {
                            cropSize = drawW / targetAspect
                        } else {
                            cropSize = drawH
                        }
                    }
                    val cw =
                        if (targetAspect >= 1f) cropSize else cropSize * targetAspect
                    val ch =
                        if (targetAspect >= 1f) cropSize / targetAspect else cropSize
                    cropLeft = drawLeft + (drawW - cw) / 2f
                    cropTop = drawTop + (drawH - ch) / 2f
                }

                val cw = if (targetAspect >= 1f) cropSize else cropSize * targetAspect
                val ch = if (targetAspect >= 1f) cropSize / targetAspect else cropSize

                // Dim area outside the crop
                val dimColor = Color.Black.copy(alpha = 0.5f)
                // Top
                drawRect(dimColor, Offset(drawLeft, drawTop), Size(drawW, cropTop - drawTop))
                // Bottom
                drawRect(dimColor, Offset(drawLeft, cropTop + ch), Size(drawW, drawTop + drawH - cropTop - ch))
                // Left
                drawRect(dimColor, Offset(drawLeft, cropTop), Size(cropLeft - drawLeft, ch))
                // Right
                drawRect(dimColor, Offset(cropLeft + cw, cropTop), Size(drawLeft + drawW - cropLeft - cw, ch))

                // Crop border
                drawRect(
                    color = Color.White,
                    topLeft = Offset(cropLeft, cropTop),
                    size = Size(cw, ch),
                    style = Stroke(width = 3f)
                )

                // Rule of thirds grid lines
                val gridColor = Color.White.copy(alpha = 0.4f)
                for (i in 1..2) {
                    val xLine = cropLeft + cw * i / 3f
                    drawLine(gridColor, Offset(xLine, cropTop), Offset(xLine, cropTop + ch), strokeWidth = 1f)
                    val yLine = cropTop + ch * i / 3f
                    drawLine(gridColor, Offset(cropLeft, yLine), Offset(cropLeft + cw, yLine), strokeWidth = 1f)
                }
            }
        }
    }
}

private fun bitmapToRawRgbBase64(bitmap: Bitmap, width: Int, height: Int): String {
    val pixels = IntArray(width * height)
    bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
    val rgb = ByteArray(width * height * 3)
    for (i in pixels.indices) {
        rgb[i * 3] = ((pixels[i] shr 16) and 0xFF).toByte()
        rgb[i * 3 + 1] = ((pixels[i] shr 8) and 0xFF).toByte()
        rgb[i * 3 + 2] = (pixels[i] and 0xFF).toByte()
    }
    return Base64.getEncoder().encodeToString(rgb)
}
