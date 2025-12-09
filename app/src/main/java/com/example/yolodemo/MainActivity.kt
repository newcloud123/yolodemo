package com.example.yolodemo

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.drawable.Drawable
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.ByteArrayOutputStream // <--- 关键修复：添加了这行引用
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

// ================= YoloV8Detector 类 (保持不变) =================
class YoloV8Detector(
    private val context: Context,
    private val modelPath: String = "yolov8n_float32.tflite",
    private val labelPath: String = "labels.txt"
) {
    private var interpreter: Interpreter? = null
    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0
    private val labels = mutableListOf<String>()

    private val imageProcessor by lazy {
        ImageProcessor.Builder()
            .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
            .add(CastOp(INPUT_IMAGE_TYPE))
            .build()
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = org.tensorflow.lite.DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = org.tensorflow.lite.DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F
        private const val BOX_STROKE_WIDTH = 8f
        private const val TEXT_SIZE = 40f
    }

    data class BoundingBox(
        val x1: Float, val y1: Float, val x2: Float, val y2: Float,
        val cx: Float, val cy: Float, val w: Float, val h: Float,
        val cnf: Float, val cls: Int, val clsName: String
    )

    init {
        initInterpreter()
        loadLabels()
    }

    private fun initInterpreter() {
        try {
            val modelBuffer = loadModelFile()
            val options = Interpreter.Options().apply {
                numThreads = 4
                setUseNNAPI(false)
            }
            interpreter = Interpreter(modelBuffer, options)
            val inputShape = interpreter?.getInputTensor(0)?.shape()
            val outputShape = interpreter?.getOutputTensor(0)?.shape()
            tensorHeight = inputShape?.get(1) ?: 0
            tensorWidth = inputShape?.get(2) ?: 0
            numChannel = outputShape?.get(1) ?: 0
            numElements = outputShape?.get(2) ?: 0
        } catch (e: Exception) {
            e.printStackTrace()
            throw RuntimeException("初始化TFLite解释器失败: ${e.message}")
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        try {
            val assetManager = context.assets
            val assetFileDescriptor = assetManager.openFd(modelPath)
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, assetFileDescriptor.startOffset, assetFileDescriptor.declaredLength)
        } catch (e: Exception) {
            throw RuntimeException("加载模型文件失败", e)
        }
    }

    private fun loadLabels() {
        try {
            context.assets.open(labelPath).use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        line?.takeIf { it.isNotEmpty() }?.let { labels.add(it) }
                    }
                }
            }
        } catch (e: IOException) {
            throw RuntimeException("加载标签文件失败", e)
        }
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, tensorWidth, tensorHeight, true)
        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        return processedImage.buffer.apply { order(ByteOrder.nativeOrder()) }
    }

    fun infer(bitmap: Bitmap): List<BoundingBox>? {
        val interpreter = this.interpreter ?: return null
        val inputBuffer = preprocess(bitmap)
        val outputShape = intArrayOf(1, numChannel, numElements)
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, OUTPUT_IMAGE_TYPE)
        interpreter.run(inputBuffer, outputBuffer.buffer)
        return bestBox(outputBuffer.floatArray)
    }

    private fun bestBox(outputArray: FloatArray): List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()
        for (c in 0 until numElements) {
            var maxConf = -1.0f
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel) {
                if (outputArray[arrayIdx] > maxConf) {
                    maxConf = outputArray[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }
            if (maxConf <= CONFIDENCE_THRESHOLD || maxIdx < 0 || maxIdx >= labels.size) continue
            val cx = outputArray[c]
            val cy = outputArray[c + numElements]
            val w = outputArray[c + numElements * 2]
            val h = outputArray[c + numElements * 3]
            val x1 = cx - (w / 2F)
            val y1 = cy - (h / 2F)
            val x2 = cx + (w / 2F)
            val y2 = cy + (h / 2F)
            if (x1 < 0F || x1 > 1F || y1 < 0F || y1 > 1F || x2 < 0F || x2 > 1F || y2 < 0F || y2 > 1F) continue
            boundingBoxes.add(BoundingBox(x1, y1, x2, y2, cx, cy, w, h, maxConf, maxIdx, labels[maxIdx]))
        }
        return if (boundingBoxes.isEmpty()) null else applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>): MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()
        while (sortedBoxes.isNotEmpty()) {
            val currentBox = sortedBoxes.first()
            selectedBoxes.add(currentBox)
            sortedBoxes.remove(currentBox)
            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                if (calculateIoU(currentBox, iterator.next()) >= IOU_THRESHOLD) iterator.remove()
            }
        }
        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val intersectX1 = max(box1.x1, box2.x1)
        val intersectY1 = max(box1.y1, box2.y1)
        val intersectX2 = min(box1.x2, box2.x2)
        val intersectY2 = min(box1.y2, box2.y2)
        val intersectionArea = max(0F, intersectX2 - intersectX1) * max(0F, intersectY2 - intersectY1)
        if (intersectionArea == 0F) return 0F
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    fun drawBoundingBoxes(bitmap: Bitmap, boxes: List<BoundingBox>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val boxPaint = Paint().apply { color = Color.RED; style = Paint.Style.STROKE; strokeWidth = BOX_STROKE_WIDTH; isAntiAlias = true }
        val bgPaint = Paint().apply { color = Color.BLACK; alpha = 180; isAntiAlias = true }
        val textPaint = Paint().apply { color = Color.WHITE; textSize = TEXT_SIZE; typeface = Typeface.DEFAULT_BOLD; isAntiAlias = true }

        for (box in boxes) {
            val rectF = RectF(box.x1 * mutableBitmap.width, box.y1 * mutableBitmap.height, box.x2 * mutableBitmap.width, box.y2 * mutableBitmap.height)
            canvas.drawRect(rectF, boxPaint)
            val text = "${box.clsName} ${String.format("%.2f", box.cnf)}"
            val textBounds = Rect()
            textPaint.getTextBounds(text, 0, text.length, textBounds)
            val bgRect = RectF(rectF.left, rectF.top - textBounds.height() - 10, rectF.left + textBounds.width() + 20, rectF.top)
            canvas.drawRoundRect(bgRect, 10f, 10f, bgPaint)
            canvas.drawText(text, rectF.left + 10, rectF.top - 5, textPaint)
        }
        return mutableBitmap
    }

    fun release() {
        interpreter?.close()
        interpreter = null
        labels.clear()
    }
}

// ================= MainActivity =================
class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var btnCamera: Button
    private lateinit var btnDetect: Button
    private lateinit var tvFps: TextView // FPS 显示控件

    private lateinit var yoloDetector: YoloV8Detector
    private var isCameraProviderReady = false
    private var cameraProvider: ProcessCameraProvider? = null

    // FPS 计算相关
    private var lastTimestamp = 0L
    private var frameCount = 0
    private val FPS_WINDOW_SIZE = 5

    // 关键状态标志
    @Volatile // 确保多线程可见性
    private var detectRunning = false
    private var cameraRunning = false

    private val preview: Preview by lazy {
        Preview.Builder().build().apply {
            setSurfaceProvider(previewView.surfaceProvider)
        }
    }

    private val imageAnalysis: ImageAnalysis by lazy {
        ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .apply {
                setAnalyzer(cameraExecutor) { imageProxy ->
                    processImageProxy(imageProxy)
                }
            }
    }

    private var camera: Camera? = null
    private lateinit var cameraExecutor: ExecutorService

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) initCameraProvider()
            else Toast.makeText(this, "需要摄像头权限", Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 初始化 YOLO
        try {
            yoloDetector = YoloV8Detector(this)
        } catch (e: Exception) {
            Toast.makeText(this, "模型加载失败：${e.message}", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        previewView = findViewById(R.id.previewView)
        btnCamera = findViewById(R.id.btnCamera)
        btnDetect = findViewById(R.id.btnDetect)
        tvFps = findViewById(R.id.tvFps)

        // 关键设置：使用 FILL_CENTER 确保画面铺满全屏（配合 ResultDrawable 的 max 逻辑）
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

        btnCamera.isEnabled = false
        btnDetect.isEnabled = false

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) initCameraProvider()
        else requestPermissionLauncher.launch(Manifest.permission.CAMERA)

        btnCamera.setOnClickListener {
            if (cameraRunning) stopCamera() else startCamera()
        }

        btnDetect.setOnClickListener {
            if (detectRunning) stopDetect() else startDetect()
        }
    }

    // ================= 核心修复：ResultDrawable =================
    private inner class ResultDrawable(private val bitmap: Bitmap) : Drawable() {
        override fun getOpacity(): Int = PixelFormat.OPAQUE

        // 全黑背景画笔
        private val clearPaint = Paint().apply {
            color = Color.BLACK
            style = Paint.Style.FILL
        }

        override fun draw(canvas: Canvas) {
            // 1. 绘制全黑背景，遮挡底层 Preview
            canvas.drawRect(0f, 0f, canvas.width.toFloat(), canvas.height.toFloat(), clearPaint)

            val viewWidth = previewView.width.toFloat()
            val viewHeight = previewView.height.toFloat()
            val imgWidth = bitmap.width.toFloat()
            val imgHeight = bitmap.height.toFloat()

            // 关键修复 (Issue 2): 使用 max() 而不是 min()
            // FILL_CENTER 模式下，图像是“裁切填充”的，所以要用较大的比例来铺满屏幕
            val scaleX = viewWidth / imgWidth
            val scaleY = viewHeight / imgHeight
            val scale = max(scaleX, scaleY)

            // 计算居中偏移（对于 FILL_CENTER，这里的 offset 可能是负数，这是正确的）
            val dx = (viewWidth - imgWidth * scale) / 2
            val dy = (viewHeight - imgHeight * scale) / 2

            val matrix = Matrix()
            matrix.postScale(scale, scale)
            matrix.postTranslate(dx, dy)

            canvas.drawBitmap(bitmap, matrix, Paint().apply { isAntiAlias = true })
        }

        override fun setAlpha(alpha: Int) {}
        override fun setColorFilter(colorFilter: ColorFilter?) {}
    }

    // ================= 核心修复：处理帧逻辑 =================
    private fun processImageProxy(imageProxy: ImageProxy) {
        // 第一道防线
        if (!detectRunning) {
            imageProxy.close()
            return
        }

        val currentTime = System.currentTimeMillis()

        try {
            val bitmap = imageProxy.toBitmap() ?: return
            val boxes = yoloDetector.infer(bitmap)
            val resultBitmap = boxes?.let { yoloDetector.drawBoundingBoxes(bitmap, it) } ?: bitmap

            // FPS 计算
            frameCount++
            if (frameCount >= FPS_WINDOW_SIZE) {
                if (lastTimestamp != 0L) {
                    val duration = currentTime - lastTimestamp
                    if (duration > 0) {
                        val fps = frameCount * 1000.0 / duration
                        runOnUiThread { tvFps.text = String.format("FPS: %.1f", fps) }
                    }
                }
                lastTimestamp = currentTime
                frameCount = 0
            }

            // 更新 UI
            runOnUiThread {
                // 第二道防线 (Issue 1 修复关键):
                // 再次检查 detectRunning。防止在点击停止瞬间，后台线程刚跑完，
                // 却把黑色不透明的图层加回去了，导致屏幕卡在最后一帧。
                if (!detectRunning) {
                    previewView.overlay.clear()
                    return@runOnUiThread
                }

                previewView.overlay.clear()
                previewView.overlay.add(ResultDrawable(resultBitmap))
            }

        } catch (e: Exception) {
            Log.e("ImageAnalysis", "错误: ${e.message}", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun ImageProxy.toBitmap(): Bitmap? {
        val image = this.image ?: return null
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val outputStream = ByteArrayOutputStream() // <--- 现在这行代码可以正常工作了
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, outputStream)
        val jpegData = outputStream.toByteArray()
        val originalBitmap = BitmapFactory.decodeByteArray(jpegData, 0, jpegData.size)
        val matrix = Matrix().apply {
            postRotate(this@toBitmap.imageInfo.rotationDegrees.toFloat())
            // 后置摄像头通常不需要镜像，前置需要。这里保持默认不镜像。
        }
        return Bitmap.createBitmap(originalBitmap, 0, 0, originalBitmap.width, originalBitmap.height, matrix, true)
    }

    private fun initCameraProvider() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                cameraProvider = providerFuture.get()
                isCameraProviderReady = true
                btnCamera.isEnabled = true
            } catch (e: Exception) {
                Log.e("CameraX", "初始化失败", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startCamera() {
        if (!isCameraProviderReady) return
        val provider = cameraProvider ?: return
        try {
            provider.unbindAll()
            camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview)
            cameraRunning = true
            btnCamera.text = "关闭摄像头"
            btnDetect.isEnabled = true
        } catch (e: Exception) {
            Toast.makeText(this, "相机启动失败", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopCamera() {
        cameraProvider?.unbindAll()
        cameraRunning = false
        detectRunning = false // 级联停止
        btnCamera.text = "打开摄像头"
        btnDetect.text = "开始识别"
        btnDetect.isEnabled = false
        previewView.overlay.clear()
        tvFps.text = "FPS: 0.0"
    }

    private fun startDetect() {
        if (!cameraRunning) return
        val provider = cameraProvider ?: return
        try {
            provider.unbindAll()
            // 绑定 Preview 和 ImageAnalysis
            camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis)
            detectRunning = true
            btnDetect.text = "停止识别"
            frameCount = 0
            lastTimestamp = 0
        } catch (e: Exception) {
            Log.e("Detect", "启动识别失败", e)
        }
    }

    private fun stopDetect() {
        if (!cameraRunning) return
        val provider = cameraProvider ?: return

        // 1. 先设置标志位，阻止后续帧处理
        detectRunning = false
        btnDetect.text = "开始识别"
        tvFps.text = "FPS: 0.0"

        // 2. 立即清空 Overlay (防止当前帧残留)
        previewView.overlay.clear()

        try {
            provider.unbindAll()
            // 3. 重新绑定仅 Preview，恢复原始流
            camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview)
        } catch (e: Exception) {
            Log.e("Detect", "停止识别失败", e)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        yoloDetector.release()
        cameraExecutor.shutdown()
        cameraProvider?.unbindAll()
    }

    private fun allPermissionsGranted() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
}