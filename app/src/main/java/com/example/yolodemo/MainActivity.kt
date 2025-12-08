package com.example.yolodemo

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.camera.view.PreviewView
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.ExecutionException

class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var btnCamera: Button
    private lateinit var btnDetect: Button

    // 新增：标记 CameraProvider 是否初始化完成
    private var isCameraProviderReady = false
    private var cameraProvider: ProcessCameraProvider? = null
    private val preview: Preview by lazy {
        Preview.Builder().build().apply {
            setSurfaceProvider(previewView.surfaceProvider)
        }
    }
    private val imageAnalysis: ImageAnalysis by lazy {
        ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .apply {
                setAnalyzer(cameraExecutor) { imageProxy ->
                    Log.d("Detect", "分析一帧: ${imageProxy.width}x${imageProxy.height}")
                    imageProxy.close()
                }
            }
    }

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
            Log.d("Detect", "识别功能已停止")
        } catch (e: Exception) {
            Log.e("Detect", "停止识别失败: ${e.message}", e)
            Toast.makeText(this, "停止识别失败：${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        cameraProvider?.unbindAll()
    }

    /**
     * 检查摄像头权限
     */
    private fun allPermissionsGranted() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
}