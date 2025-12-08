package com.example.yolodemo

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.ExecutionException
import kotlin.math.max
import kotlin.math.min
import android.content.Context
import android.graphics.drawable.Drawable
/**
 * YOLOv8 TFLite推理类
 * @param context 上下文
 * @param modelPath assets中的模型文件路径（如"yolov8s_float32.tflite"）
 * @param labelPath assets中的标签文件路径（如"labels.txt"）
 */
class YoloV8Detector(
    private val context: Context,
    private val modelPath: String = "yolov8n_float32.tflite",
    private val labelPath: String = "labels.txt"
) {
    // TFLite解释器
    private var interpreter: Interpreter? = null

    // 模型输入输出维度
    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    // 标签列表
    private val labels = mutableListOf<String>()

    // 图像预处理处理器
    private val imageProcessor by lazy {
        ImageProcessor.Builder()
            .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
            .add(CastOp(INPUT_IMAGE_TYPE))
            .build()
    }

    companion object {
        // 预处理常量
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = org.tensorflow.lite.DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = org.tensorflow.lite.DataType.FLOAT32

        // 后处理阈值
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F

        // 绘制相关常量
        private const val BOX_STROKE_WIDTH = 8f
        private const val TEXT_SIZE = 40f
    }

    /**
     * 检测框数据类
     * @param x1 左上角x（归一化0-1）
     * @param y1 左上角y（归一化0-1）
     * @param x2 右下角x（归一化0-1）
     * @param y2 右下角y（归一化0-1）
     * @param cx 中心x（归一化0-1）
     * @param cy 中心y（归一化0-1）
     * @param w 宽度（归一化0-1）
     * @param h 高度（归一化0-1）
     * @param cnf 置信度
     * @param cls 类别索引
     * @param clsName 类别名称
     */
    data class BoundingBox(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val cx: Float,
        val cy: Float,
        val w: Float,
        val h: Float,
        val cnf: Float,
        val cls: Int,
        val clsName: String
    )

    init {
        // 初始化模型和标签
        initInterpreter()
        loadLabels()
    }

    /**
     * 初始化TFLite解释器
     */
    private fun initInterpreter() {
        try {
            // 加载模型文件（内存映射方式，效率更高）
            try {
                val modelBuffer = loadModelFile()
            }
            catch (e: Exception) {
                e.printStackTrace()
                throw RuntimeException("加载模型失败: ${e.message}")
            }
            val modelBuffer = loadModelFile()
            val options = Interpreter.Options().apply {
                numThreads = 4  // 设置线程数
                setUseNNAPI(false)  // 可选：开启NNAPI加速（需设备支持）
            }
            interpreter = Interpreter(modelBuffer, options)

            // 获取输入输出张量形状
            val inputShape = interpreter?.getInputTensor(0)?.shape()
            val outputShape = interpreter?.getOutputTensor(0)?.shape()

            // 解析维度（YOLOv8输入形状：[1, H, W, 3] 或 [1, W, H, 3]，需根据实际模型调整）
            tensorHeight = inputShape?.get(1) ?: 0
            tensorWidth = inputShape?.get(2) ?: 0

            // 解析输出维度（YOLOv8输出形状：[1, numChannel, numElements]）
            numChannel = outputShape?.get(1) ?: 0
            numElements = outputShape?.get(2) ?: 0

            if (tensorWidth == 0 || tensorHeight == 0 || numChannel == 0 || numElements == 0) {
                throw RuntimeException("解析模型输入输出维度失败")
            }
        } catch (e: Exception) {
            e.printStackTrace()
            throw RuntimeException("初始化TFLite解释器失败: ${e.message}")
        }
    }

    /**
     * 加载assets中的模型文件为MappedByteBuffer
     */
    private fun loadModelFile(): MappedByteBuffer {
        // 1. 打印待加载的模型路径（排查路径错误）
        Log.d("YoloV8Detector", "开始加载模型文件，路径：$modelPath")

        try {
            // 2. 主动检查assets中是否存在该模型文件（提前预判文件不存在问题）
            val assetManager = context.assets
            val assetFiles = assetManager.list("") ?: emptyArray() // 列出assets根目录文件
            val isModelExist = assetFiles.contains(modelPath)
            if (isModelExist) {
                Log.d("YoloV8Detector", "✅ 模型文件存在：$modelPath")
            } else {
                Log.e("YoloV8Detector", "❌ assets根目录未找到模型文件：$modelPath")
                // 可选：列出所有assets文件，方便排查路径错误
                Log.d("YoloV8Detector", "assets根目录所有文件：${assetFiles.joinToString(", ")}")
            }

            // 3. 打开文件描述符并打印关键信息
            val assetFileDescriptor = assetManager.openFd(modelPath)
            Log.d("YoloV8Detector", "模型文件描述符信息：")
            Log.d("YoloV8Detector", "  - 起始偏移量：${assetFileDescriptor.startOffset}")
            Log.d("YoloV8Detector", "  - 文件长度：${assetFileDescriptor.declaredLength} 字节")
            Log.d("YoloV8Detector", "  - 文件总长度：${assetFileDescriptor.length} 字节")

            // 4. 打开文件流并映射内存
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength

            // 5. 内存映射并打印成功日志
            val mappedBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            Log.d("YoloV8Detector", "✅ 模型文件加载成功！映射内存大小：${mappedBuffer.capacity()} 字节")

            return mappedBuffer
        } catch (e: FileNotFoundException) {
            // 捕获文件不存在异常（最常见错误）
            Log.e("YoloV8Detector", "❌ 加载模型失败：文件不存在 - $modelPath", e)
            throw RuntimeException("模型文件不存在：$modelPath，请检查assets目录是否包含该文件", e)
        } catch (e: IOException) {
            // 捕获其他IO异常（如权限、文件损坏等）
            Log.e("YoloV8Detector", "❌ 加载模型失败：IO异常 - $modelPath", e)
            throw RuntimeException("加载模型文件IO异常：${e.message}", e)
        } catch (e: Exception) {
            // 捕获其他未知异常
            Log.e("YoloV8Detector", "❌ 加载模型失败：未知异常 - $modelPath", e)
            throw RuntimeException("加载模型文件失败：${e.message}", e)
        }
    }

    /**
     * 加载标签文件
     */
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
            e.printStackTrace()
            throw RuntimeException("加载标签文件失败: ${e.message}")
        }
    }

    /**
     * 图像预处理：缩放+归一化+类型转换
     * @param bitmap 原始图像
     * @return 处理后的输入Buffer
     */
    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        // 缩放图像到模型输入尺寸
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, tensorWidth, tensorHeight, true)

        // 转换为TensorImage并预处理
        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)

        return processedImage.buffer.apply {
            order(ByteOrder.nativeOrder())  // 确保字节序正确
        }
    }

    /**
     * 执行推理
     * @param bitmap 原始图像
     * @return 经过NMS后的检测框列表（null表示无检测结果）
     */
    fun infer(bitmap: Bitmap): List<BoundingBox>? {
        // 检查解释器是否初始化
        val interpreter = this.interpreter ?: return null

        // 1. 预处理
        val inputBuffer = preprocess(bitmap)

        // 2. 准备输出缓冲区
        val outputShape = intArrayOf(1, numChannel, numElements)
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, OUTPUT_IMAGE_TYPE)

        // 3. 执行推理
        interpreter.run(inputBuffer, outputBuffer.buffer)

        // 4. 后处理（提取有效框 + NMS）
        return bestBox(outputBuffer.floatArray)
    }

    /**
     * 提取置信度符合要求的检测框
     */
    private fun bestBox(outputArray: FloatArray): List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            var maxConf = -1.0f
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j

            // 找到当前位置置信度最高的类别
            while (j < numChannel) {
                if (outputArray[arrayIdx] > maxConf) {
                    maxConf = outputArray[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            // 过滤低置信度框
            if (maxConf <= CONFIDENCE_THRESHOLD || maxIdx < 0 || maxIdx >= labels.size) {
                continue
            }

            // 解析框坐标（YOLOv8输出为归一化坐标）
            val cx = outputArray[c] // 中心x
            val cy = outputArray[c + numElements] // 中心y
            val w = outputArray[c + numElements * 2] // 宽度
            val h = outputArray[c + numElements * 3] // 高度

            // 转换为左上角/右下角坐标
            val x1 = cx - (w / 2F)
            val y1 = cy - (h / 2F)
            val x2 = cx + (w / 2F)
            val y2 = cy + (h / 2F)

            // 过滤超出边界的框
            if (x1 < 0F || x1 > 1F || y1 < 0F || y1 > 1F || x2 < 0F || x2 > 1F || y2 < 0F || y2 > 1F) {
                continue
            }

            // 添加有效框
            boundingBoxes.add(
                BoundingBox(
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    cx = cx,
                    cy = cy,
                    w = w,
                    h = h,
                    cnf = maxConf,
                    cls = maxIdx,
                    clsName = labels[maxIdx]
                )
            )
        }

        return if (boundingBoxes.isEmpty()) null else applyNMS(boundingBoxes)
    }

    /**
     * 非极大值抑制（NMS）
     */
    private fun applyNMS(boxes: List<BoundingBox>): MutableList<BoundingBox> {
        // 按置信度降序排序
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while (sortedBoxes.isNotEmpty()) {
            // 取置信度最高的框
            val currentBox = sortedBoxes.first()
            selectedBoxes.add(currentBox)
            sortedBoxes.remove(currentBox)

            // 移除与当前框IOU超过阈值的框
            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val box = iterator.next()
                if (calculateIoU(currentBox, box) >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }

    /**
     * 计算两个框的交并比（IOU）
     */
    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        // 计算交集区域
        val intersectX1 = max(box1.x1, box2.x1)
        val intersectY1 = max(box1.y1, box2.y1)
        val intersectX2 = min(box1.x2, box2.x2)
        val intersectY2 = min(box1.y2, box2.y2)

        // 交集面积
        val intersectionArea = max(0F, intersectX2 - intersectX1) * max(0F, intersectY2 - intersectY1)
        if (intersectionArea == 0F) return 0F

        // 两个框的面积
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h

        // 计算IOU
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    /**
     * 在图像上绘制检测框和标签
     * @param bitmap 原始图像
     * @param boxes 检测框列表
     * @return 绘制后的图像
     */
    fun drawBoundingBoxes(bitmap: Bitmap, boxes: List<BoundingBox>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)

        // 框的画笔
        val boxPaint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = BOX_STROKE_WIDTH
            isAntiAlias = true
        }

        // 文字背景画笔
        val bgPaint = Paint().apply {
            color = Color.BLACK
            alpha = 180
            isAntiAlias = true
        }

        // 文字画笔
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = TEXT_SIZE
            typeface = Typeface.DEFAULT_BOLD
            isAntiAlias = true
        }

        // 绘制每个检测框
        for (box in boxes) {
            // 转换归一化坐标到图像像素坐标
            val rectF = RectF(
                box.x1 * mutableBitmap.width,
                box.y1 * mutableBitmap.height,
                box.x2 * mutableBitmap.width,
                box.y2 * mutableBitmap.height
            )

            // 绘制框
            canvas.drawRect(rectF, boxPaint)

            // 绘制文字背景（避免文字与图像融合）
            val text = "${box.clsName} ${String.format("%.2f", box.cnf)}"
            val textBounds = Rect()
            textPaint.getTextBounds(text, 0, text.length, textBounds)
            val bgRect = RectF(
                rectF.left,
                rectF.top - textBounds.height() - 10,
                rectF.left + textBounds.width() + 20,
                rectF.top
            )
            canvas.drawRoundRect(bgRect, 10f, 10f, bgPaint)

            // 绘制文字
            canvas.drawText(text, rectF.left + 10, rectF.top - 5, textPaint)
        }

        return mutableBitmap
    }

    /**
     * 释放资源（建议在Activity/Fragment销毁时调用）
     */
    fun release() {
        interpreter?.close()
        interpreter = null
        labels.clear()
    }
}

class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var btnCamera: Button
    private lateinit var btnDetect: Button

    // YOLO检测器实例
    private lateinit var yoloDetector: YoloV8Detector

    // 新增：标记 CameraProvider 是否初始化完成
    private var isCameraProviderReady = false
    private var cameraProvider: ProcessCameraProvider? = null
    private val preview: Preview by lazy {
        Preview.Builder().build().apply {
            setSurfaceProvider(previewView.surfaceProvider)
        }
    }

    // 图像分析器（集成YOLO推理）
    private val imageAnalysis: ImageAnalysis by lazy {
        ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888) // 简化Bitmap转换
            .build()
            .apply {
                setAnalyzer(cameraExecutor) { imageProxy ->
                    processImageProxy(imageProxy)
                }
            }
    }

    // 明确指定为CameraX的Camera类，解决歧义
    private var camera: Camera? = null
    private var cameraRunning = false
    private var detectRunning = false
    private lateinit var cameraExecutor: ExecutorService

    // 权限申请回调
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                initCameraProvider()
            } else {
                // 权限被拒绝：引导用户去设置页开启
                val permissionDeniedMsg = "摄像头权限被拒绝，无法使用功能！请前往设置开启权限。"
                Toast.makeText(this, permissionDeniedMsg, Toast.LENGTH_LONG).show()
                Log.e("Permission", permissionDeniedMsg)

                // 跳转到应用设置页面
                val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                    data = Uri.fromParts("package", packageName, null)
                }
                startActivity(intent)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 初始化YOLO检测器
        try {
            yoloDetector = YoloV8Detector(this)
        } catch (e: Exception) {
            Log.e("YOLO", "初始化YOLO检测器失败: ${e.message}", e)
            Toast.makeText(this, "YOLO模型加载失败：${e.message}", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        previewView = findViewById(R.id.previewView)
        btnCamera = findViewById(R.id.btnCamera)
        btnDetect = findViewById(R.id.btnDetect)

        // 初始时禁用按钮，避免用户误点
        btnCamera.isEnabled = false
        btnDetect.isEnabled = false

        cameraExecutor = Executors.newSingleThreadExecutor()

        // 权限检查
        if (allPermissionsGranted()) {
            initCameraProvider()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        btnCamera.setOnClickListener {
            if (cameraRunning) stopCamera()
            else startCamera()
        }

        btnDetect.setOnClickListener {
            if (detectRunning) stopDetect()
            else startDetect()
        }
    }
    private inner class ResultDrawable(private val bitmap: Bitmap) : Drawable() {
        override fun draw(canvas: Canvas) {
            // 缩放Bitmap到PreviewView尺寸
            val matrix = Matrix()
            val scaleX = previewView.width.toFloat() / bitmap.width
            val scaleY = previewView.height.toFloat() / bitmap.height
            val scale = max(scaleX, scaleY)
            matrix.postScale(scale, scale)
            matrix.postTranslate(
                (previewView.width - bitmap.width * scale) / 2,
                (previewView.height - bitmap.height * scale) / 2
            )
            canvas.drawBitmap(bitmap, matrix, Paint().apply { isAntiAlias = true })
        }

        override fun setAlpha(alpha: Int) {}
        override fun getOpacity(): Int = PixelFormat.TRANSLUCENT
        override fun setColorFilter(colorFilter: ColorFilter?) {}
    }
    /**
     * 处理CameraX的ImageProxy帧，执行YOLO推理并绘制结果
     */
    private fun processImageProxy(imageProxy: ImageProxy) {
        try {
            // 1. 将ImageProxy转换为Bitmap（注意：必须在imageProxy.close()前完成）
            val bitmap = imageProxy.toBitmap() ?: return

            // 2. 执行YOLO推理
            val boxes = yoloDetector.infer(bitmap)

            // 3. 绘制检测框（如果有检测结果）
            val resultBitmap = boxes?.let { yoloDetector.drawBoundingBoxes(bitmap, it) } ?: bitmap

            // 4. 在PreviewView上显示结果（修复add类型歧义）
            runOnUiThread {
                previewView.overlay.clear()
                val resultDrawable = ResultDrawable(resultBitmap)
                previewView.overlay.add(resultDrawable)
            }

        } catch (e: Exception) {
            Log.e("ImageAnalysis", "处理帧失败: ${e.message}", e)
        } finally {
            // 必须关闭ImageProxy，否则会阻塞后续帧
            imageProxy.close()
        }
    }

    /**
     * 扩展函数：ImageProxy转Bitmap（RGBA格式）
     */
    // 修复后的ImageProxy转Bitmap
    private fun ImageProxy.toBitmap(): Bitmap? {
        val image = this.image ?: return null
        val yBuffer = image.planes[0].buffer // Y通道
        val uBuffer = image.planes[1].buffer // U通道
        val vBuffer = image.planes[2].buffer // V通道

        // 转换NV21格式
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        // NV21转Bitmap
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val outputStream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, outputStream)
        val jpegData = outputStream.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(jpegData, 0, jpegData.size)

        // 纠正旋转和后置摄像头镜像
        val matrix = Matrix().apply {
            postRotate(this@toBitmap.imageInfo.rotationDegrees.toFloat())
            if (camera?.cameraInfo?.lensFacing == CameraSelector.LENS_FACING_BACK) {
                postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
            }
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    // 新增：具名Drawable内部类（解决add类型歧义）

    /**
     * 初始化 CameraProvider，完成后启用按钮
     */
    private fun initCameraProvider() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                cameraProvider = providerFuture.get()
                isCameraProviderReady = true
                // 初始化完成后启用按钮
                btnCamera.isEnabled = true
                btnDetect.isEnabled = false // 初始仅启用摄像头按钮
                Log.d("CameraX", "CameraProvider 初始化完成，按钮已启用")
            } catch (e: ExecutionException) {
                Log.e("CameraX", "初始化 CameraProvider 失败: ${e.message}", e)
                Toast.makeText(this, "摄像头初始化失败：${e.message}", Toast.LENGTH_SHORT).show()
            } catch (e: InterruptedException) {
                Log.e("CameraX", "初始化 CameraProvider 被中断: ${e.message}", e)
                Toast.makeText(this, "摄像头初始化被中断", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * 启动摄像头（仅 Preview）
     */
    private fun startCamera() {
        if (!isCameraProviderReady) {
            val tip = "摄像头初始化中，请稍等！"
            Toast.makeText(this, tip, Toast.LENGTH_SHORT).show()
            Log.e("CameraX", tip)
            return
        }
        val provider = cameraProvider ?: return

        val selector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            provider.unbindAll()
            camera = provider.bindToLifecycle(this, selector, preview)
            cameraRunning = true
            btnCamera.text = "关闭摄像头"
            btnDetect.isEnabled = true // 摄像头启动后启用识别按钮
            Log.d("CameraX", "摄像头启动成功")
        } catch (e: Exception) {
            Log.e("CameraX", "启动摄像头失败: ${e.message}", e)
            Toast.makeText(this, "启动摄像头失败：${e.message}", Toast.LENGTH_SHORT).show()
            cameraRunning = false
        }
    }

    /**
     * 停止摄像头（解绑所有 UseCase）
     */
    private fun stopCamera() {
        cameraProvider?.unbindAll()
        cameraRunning = false
        detectRunning = false
        btnCamera.text = "打开摄像头"
        btnDetect.text = "开始识别"
        btnDetect.isEnabled = false // 关闭摄像头后禁用识别按钮
        // 清空预览层绘制
        previewView.overlay.clear()
        Log.d("CameraX", "摄像头已关闭")
    }

    /**
     * 启动识别（绑定 Preview + ImageAnalysis）
     */
    private fun startDetect() {
        if (!cameraRunning) {
            val tip = "请先打开摄像头！"
            Toast.makeText(this, tip, Toast.LENGTH_SHORT).show()
            Log.e("Detect", tip)
            return
        }
        if (!isCameraProviderReady) return

        val provider = cameraProvider ?: return
        val selector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            provider.unbindAll()
            camera = provider.bindToLifecycle(this, selector, preview, imageAnalysis)
            detectRunning = true
            btnDetect.text = "停止识别"
            Log.d("Detect", "识别功能已启动")
        } catch (e: Exception) {
            Log.e("Detect", "启动识别失败: ${e.message}", e)
            Toast.makeText(this, "启动识别失败：${e.message}", Toast.LENGTH_SHORT).show()
            detectRunning = false
        }
    }

    /**
     * 停止识别（解绑所有，重新绑定仅 Preview）
     */
    private fun stopDetect() {
        if (!cameraRunning) return
        val provider = cameraProvider ?: return

        val selector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            provider.unbindAll()
            camera = provider.bindToLifecycle(this, selector, preview)
            detectRunning = false
            btnDetect.text = "开始识别"
            // 清空预览层绘制
            previewView.overlay.clear()
            Log.d("Detect", "识别功能已停止")
        } catch (e: Exception) {
            Log.e("Detect", "停止识别失败: ${e.message}", e)
            Toast.makeText(this, "停止识别失败：${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // 释放YOLO检测器资源
        yoloDetector.release()
        // 关闭摄像头线程
        cameraExecutor.shutdown()
        cameraProvider?.unbindAll()
        // 清空预览层
        previewView.overlay.clear()
    }

    /**
     * 检查摄像头权限
     */
    private fun allPermissionsGranted() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
}