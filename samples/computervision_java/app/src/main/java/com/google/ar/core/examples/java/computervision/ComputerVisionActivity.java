/*
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ar.core.examples.java.computervision;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.Image;
import android.net.Uri;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.CameraConfig;
import com.google.ar.core.CameraConfigFilter;
import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.Session;
import com.google.ar.core.examples.java.common.helpers.CameraPermissionHelper;
import com.google.ar.core.examples.java.common.helpers.FullScreenHelper;
import com.google.ar.core.examples.java.common.helpers.SnackbarHelper;
import com.google.ar.core.examples.java.common.helpers.TrackingStateHelper;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;
import com.google.common.util.concurrent.ListenableFuture;

import org.apache.commons.math3.filter.DefaultMeasurementModel;
import org.apache.commons.math3.filter.DefaultProcessModel;
import org.apache.commons.math3.filter.KalmanFilter;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * This is a simple example that demonstrates CPU image access with ARCore.
 */
public class ComputerVisionActivity extends AppCompatActivity implements GLSurfaceView.Renderer, SensorEventListener {
//    private static final String TAG = ComputerVisionActivity.class.getSimpleName();
    private static final String TAG = "ww";
    private static final String CAMERA_INTRINSICS_TEXT_FORMAT =
            "\tUnrotated Camera %s %s Intrinsics:\n\tFocal Length: (%.2f, %.2f)"
                    + "\n\tPrincipal Point: (%.2f, %.2f)"
                    + "\n\t%s Image Dimensions: (%d, %d)"
                    + "\n\tUnrotated Field of View: (%.2f˚, %.2f˚)"
                    + "\n\tRender frame time: %.1f ms (%.0ffps)"
                    + "\n\tCPU image frame time: %.1f ms (%.0ffps)";
    private static final float RADIANS_TO_DEGREES = (float) (180 / Math.PI);

    // This app demonstrates two approaches to obtaining image data accessible on CPU:
    // 1. Access the CPU image directly from ARCore. This approach delivers a frame without latency
    //    (if available), but currently is lower resolution than the GPU image.
    // 2. Download the texture from GPU. This approach incurs a 1-frame latency, but allows a high
    //    resolution image.
    private enum ImageAcquisitionPath {
        CPU_DIRECT_ACCESS,
        GPU_DOWNLOAD
    }

    // Select the image acquisition path here.
    private final ImageAcquisitionPath imageAcquisitionPath = ImageAcquisitionPath.CPU_DIRECT_ACCESS;

    // Multiple CPU image Resolution.
    private enum ImageResolution {
        LOW_RESOLUTION,
        MEDIUM_RESOLUTION,
        HIGH_RESOLUTION,
    }

    // Default CPU image is low resolution.
    private ImageResolution cpuResolution = ImageResolution.LOW_RESOLUTION;

    // Session management and rendering.
    private GLSurfaceView surfaceView;
    private Session session;
    private Config config;
    private boolean installRequested;
    private final SnackbarHelper messageSnackbarHelper = new SnackbarHelper();
    private CpuImageDisplayRotationHelper cpuImageDisplayRotationHelper;
    private final TrackingStateHelper trackingStateHelper = new TrackingStateHelper(this);
    private final CpuImageRenderer cpuImageRenderer = new CpuImageRenderer();
    private final EdgeDetector edgeDetector = new EdgeDetector();

    // This lock prevents changing resolution as the frame is being rendered. ARCore requires all
    // CPU images to be released before changing resolution.
    private final Object frameImageInUseLock = new Object();

    // Camera intrinsics text view.
    private TextView cameraIntrinsicsTextView;

    // The fields below are used for the GPU_DOWNLOAD image acquisition path.
    private final TextureReader textureReader = new TextureReader();
    private int gpuDownloadFrameBufferIndex = -1;

    // ARCore full resolution GL texture typically has a size of 1920 x 1080.
    private static final int TEXTURE_WIDTH = 1920;
    private static final int TEXTURE_HEIGHT = 1080;

    // We choose a lower sampling resolution.
    private static final int IMAGE_WIDTH = 1280;
    private static final int IMAGE_HEIGHT = 720;

    // For Camera Configuration APIs usage.
    private CameraConfig cpuLowResolutionCameraConfig;
    private CameraConfig cpuMediumResolutionCameraConfig;
    private CameraConfig cpuHighResolutionCameraConfig;

    private Switch cvModeSwitch;
    private boolean isCVModeOn = true;
    private Switch focusModeSwitch;

    private final FrameTimeHelper renderFrameTimeHelper = new FrameTimeHelper();
    private final FrameTimeHelper cpuImageFrameTimeHelper = new FrameTimeHelper();


    // 相机
    private PreviewView viewFinder;
    private Button startCaptureButton;

    private ImageCapture imageCapture;
    private File outputDirectory;
    private ExecutorService cameraExecutor;

    private static final String FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS";
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String[] REQUIRED_PERMISSIONS = new String[]{Manifest.permission.CAMERA};

    private ArrayList<Uri> photoUris = new ArrayList<>();


    // 传感器
    private SensorManager sensorManager;
    private Sensor accelerometerSensor;
    private Sensor magneticFieldSensor;

    // 加速度和地磁场强度，用于计算旋转矩阵
    private float[] accelerometerValues = new float[3];
    private float[] magneticFieldValues = new float[3];

    // 通过卡尔曼滤波器（一种传感器融合算法）计算位移
    // 如果单纯使用加速度积分和速度积分，则忽略了累计误差，漂移等问题
    private KalmanFilter kalmanFilter;

    private float[] rotationMatrix = new float[9];   // 绝对角度，具体可以百度
    private float[] translationMatrix = new float[16];   // 相对于滤波器初始状态的位移
    /*
    translationMatrix:
    | 1  0  0  Tx |
    | 0  1  0  Ty |
    | 0  0  1  Tz |
    | 0  0  0  1  |
    其中，Tx, Ty, Tz 是位移。可以和 (x, y, z, 1) 相乘计算位置。
     */
    private ArrayList<float[]> rotationMatrices = new ArrayList<>();
    private ArrayList<float[]> translationMatrices = new ArrayList<>();


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        surfaceView = findViewById(R.id.surfaceview);
        cameraIntrinsicsTextView = findViewById(R.id.camera_intrinsics_view);
        surfaceView = findViewById(R.id.surfaceview);
        cvModeSwitch = (Switch) findViewById(R.id.switch_cv_mode);
        cvModeSwitch.setOnCheckedChangeListener(this::onCVModeChanged);
        focusModeSwitch = (Switch) findViewById(R.id.switch_focus_mode);
        focusModeSwitch.setOnCheckedChangeListener(this::onFocusModeChanged);
        startCaptureButton = findViewById(R.id.startCaptureButton);
        viewFinder = findViewById(R.id.viewFinder);

        cpuImageDisplayRotationHelper = new CpuImageDisplayRotationHelper(/*context=*/ this);

        // Set up renderer.
        surfaceView.setPreserveEGLContextOnPause(true);
        surfaceView.setEGLContextClientVersion(2);
        surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0); // Alpha used for plane blending.
        surfaceView.setRenderer(this);
        surfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
        surfaceView.setWillNotDraw(false);

        getLifecycle().addObserver(renderFrameTimeHelper);
        getLifecycle().addObserver(cpuImageFrameTimeHelper);

        installRequested = false;

        // 初始化传感器
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accelerometerSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        magneticFieldSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);

        // 注册传感器监听器
        sensorManager.registerListener(this, accelerometerSensor, SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(this, magneticFieldSensor, SensorManager.SENSOR_DELAY_NORMAL);

        // 初始化卡尔曼滤波器
        RealVector initialState = MatrixUtils.createRealVector(new double[]{0, 0, 0});   // 初始状态向量
        RealMatrix initialCovariance = MatrixUtils.createRealIdentityMatrix(3);   // 初始协方差矩阵

        RealMatrix processMatrix = MatrixUtils.createRealIdentityMatrix(3);   // 过程模型矩阵 A
        RealMatrix processNoiseMatrix = MatrixUtils.createRealIdentityMatrix(3);   // 过程噪声矩阵 Q
        DefaultProcessModel processModel = new DefaultProcessModel(processMatrix, null, processNoiseMatrix, initialState, initialCovariance);

        RealMatrix measurementMatrix = MatrixUtils.createRealIdentityMatrix(3);   // 测量矩阵 H
        RealMatrix measurementNoiseMatrix = MatrixUtils.createRealIdentityMatrix(3);   // 测量噪声矩阵 R
        DefaultMeasurementModel measurementModel = new DefaultMeasurementModel(measurementMatrix, measurementNoiseMatrix);

        kalmanFilter = new KalmanFilter(processModel, measurementModel);

        startCaptureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                printMatrix(rotationMatrix, 3, 3);
                printMatrix(translationMatrix, 4, 4);
                takePhoto();
            }
        });

        // 请求相机权限
        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(
                    this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }

        outputDirectory = getOutputDirectory();
        cameraExecutor = Executors.newSingleThreadExecutor();

    }

    @Override
    protected void onDestroy() {
        if (session != null) {
            // Explicitly close ARCore Session to release native resources.
            // Review the API reference for important considerations before calling close() in apps with
            // more complicated lifecycle requirements:
            // https://developers.google.com/ar/reference/java/arcore/reference/com/google/ar/core/Session#close()
            session.close();
            session = null;
        }

        cameraExecutor.shutdown();

        // 取消传感器监听器的注册，以免在应用关闭后继续消耗资源
        sensorManager.unregisterListener(this);

        super.onDestroy();
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (session == null) {
            Exception exception = null;
            String message = null;
            try {
                switch (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                    case INSTALL_REQUESTED:
                        installRequested = true;
                        return;
                    case INSTALLED:
                        break;
                }

                // ARCore requires camera permissions to operate. If we did not yet obtain runtime
                // permission on Android M and above, now is a good time to ask the user for it.
                if (!CameraPermissionHelper.hasCameraPermission(this)) {
                    CameraPermissionHelper.requestCameraPermission(this);
                    return;
                }

                session = new Session(/* context= */ this);
                config = new Config(session);
            } catch (UnavailableArcoreNotInstalledException
                     | UnavailableUserDeclinedInstallationException e) {
                message = "Please install ARCore";
                exception = e;
            } catch (UnavailableApkTooOldException e) {
                message = "Please update ARCore";
                exception = e;
            } catch (UnavailableSdkTooOldException e) {
                message = "Please update this app";
                exception = e;
            } catch (Exception e) {
                message = "This device does not support AR";
                exception = e;
            }

            if (message != null) {
                messageSnackbarHelper.showError(this, message);
                Log.e(TAG, "Exception creating session", exception);
                return;
            }
        }

        obtainCameraConfigs();

        cvModeSwitch.setChecked(cpuImageRenderer.getSplitterPosition() < 0.5f);
        focusModeSwitch.setChecked(config.getFocusMode() != Config.FocusMode.FIXED);

        // Note that order matters - see the note in onPause(), the reverse applies here.
        try {
            session.resume();
        } catch (CameraNotAvailableException e) {
            messageSnackbarHelper.showError(this, "Camera not available. Try restarting the app.");
            session = null;
            return;
        }
        surfaceView.onResume();
        cpuImageDisplayRotationHelper.onResume();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (session != null) {
            // Note that the order matters - GLSurfaceView is paused first so that it does not try
            // to query the session. If Session is paused before GLSurfaceView, GLSurfaceView may
            // still call session.update() and get a SessionPausedException.
            cpuImageDisplayRotationHelper.onPause();
            surfaceView.onPause();
            session.pause();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this,
                        "Permissions not granted by the user.",
                        Toast.LENGTH_SHORT).show();
                finish();
            }
        } else {
            if (!CameraPermissionHelper.hasCameraPermission(this)) {
                Toast.makeText(this, "Camera permission is needed to run this application", Toast.LENGTH_LONG)
                        .show();
                if (!CameraPermissionHelper.shouldShowRequestPermissionRationale(this)) {
                    // Permission denied with checking "Do not ask again".
                    CameraPermissionHelper.launchPermissionSettings(this);
                }
                finish();
            }
        }
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        FullScreenHelper.setFullScreenOnWindowFocusChanged(this, hasFocus);
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

        // Create the texture and pass it to ARCore session to be filled during update().
        try {
            cpuImageRenderer.createOnGlThread(/* context= */ this);

            // The image format can be either IMAGE_FORMAT_RGBA or IMAGE_FORMAT_I8.
            // Set keepAspectRatio to false so that the output image covers the whole viewport.
            textureReader.create(
                    /* context= */ this,
                    TextureReaderImage.IMAGE_FORMAT_I8,
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT,
                    false);

        } catch (IOException e) {
            Log.e(TAG, "Failed to read an asset file", e);
        }
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        cpuImageDisplayRotationHelper.onSurfaceChanged(width, height);
        GLES20.glViewport(0, 0, width, height);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        // Clear screen to notify driver it should not load any pixels from previous frame.
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

        if (session == null) {
            return;
        }

        // Synchronize here to avoid calling Session.update or Session.acquireCameraImage while paused.
        synchronized (frameImageInUseLock) {
            // Notify ARCore session that the view size changed so that the perspective matrix and
            // the video background can be properly adjusted.
            cpuImageDisplayRotationHelper.updateSessionIfNeeded(session);

            try {
                session.setCameraTextureName(cpuImageRenderer.getTextureId());
                final Frame frame = session.update();
                final Camera camera = frame.getCamera();

                // Keep the screen unlocked while tracking, but allow it to lock when tracking stops.
                trackingStateHelper.updateKeepScreenOnFlag(camera.getTrackingState());

                renderFrameTimeHelper.nextFrame();

                switch (imageAcquisitionPath) {
                    case CPU_DIRECT_ACCESS:
                        renderProcessedImageCpuDirectAccess(frame);
                        break;
                    case GPU_DOWNLOAD:
                        renderProcessedImageGpuDownload(frame);
                        break;
                }

                // Update the camera intrinsics' text.
                runOnUiThread(() -> cameraIntrinsicsTextView.setText(getCameraIntrinsicsText(frame)));
            } catch (Exception t) {
                // Avoid crashing the application due to unhandled exceptions.
                Log.e(TAG, "Exception on the OpenGL thread", t);
            }
        }
    }

    /* Demonstrates how to access a CPU image directly from ARCore. */
    private void renderProcessedImageCpuDirectAccess(Frame frame) {
        try (Image image = frame.acquireCameraImage()) {
            if (image.getFormat() != ImageFormat.YUV_420_888) {
                throw new IllegalArgumentException(
                        "Expected image in YUV_420_888 format, got format " + image.getFormat());
            }

            ByteBuffer processedImageBytesGrayscale = null;
            // Do not process the image with edge dectection algorithm if it is not being displayed.
            if (isCVModeOn) {
                processedImageBytesGrayscale =
                        edgeDetector.detect(
                                image.getWidth(),
                                image.getHeight(),
                                image.getPlanes()[0].getRowStride(),
                                image.getPlanes()[0].getBuffer());
            }

            cpuImageRenderer.drawWithCpuImage(
                    frame,
                    image.getWidth(),
                    image.getHeight(),
                    processedImageBytesGrayscale,
                    cpuImageDisplayRotationHelper.getViewportAspectRatio(),
                    cpuImageDisplayRotationHelper.getCameraToDisplayRotation());

            // Measure frame time since last successful execution of drawWithCpuImage().
            cpuImageFrameTimeHelper.nextFrame();
        } catch (NotYetAvailableException e) {
            // This exception will routinely happen during startup, and is expected. cpuImageRenderer
            // will handle null image properly, and will just render the background.
            cpuImageRenderer.drawWithoutCpuImage();
        }
    }

    /* Demonstrates how to access a CPU image using a download from GPU. */
    private void renderProcessedImageGpuDownload(Frame frame) {
        // If there is a frame being requested previously, acquire the pixels and process it.
        if (gpuDownloadFrameBufferIndex >= 0) {
            TextureReaderImage image = textureReader.acquireFrame(gpuDownloadFrameBufferIndex);

            if (image.format != TextureReaderImage.IMAGE_FORMAT_I8) {
                throw new IllegalArgumentException(
                        "Expected image in I8 format, got format " + image.format);
            }

            ByteBuffer processedImageBytesGrayscale =
                    edgeDetector.detect(image.width, image.height, /* stride= */ image.width, image.buffer);

            // You should always release frame buffer after using. Otherwise the next call to
            // submitFrame() may fail.
            textureReader.releaseFrame(gpuDownloadFrameBufferIndex);

            cpuImageRenderer.drawWithCpuImage(
                    frame,
                    IMAGE_WIDTH,
                    IMAGE_HEIGHT,
                    processedImageBytesGrayscale,
                    cpuImageDisplayRotationHelper.getViewportAspectRatio(),
                    cpuImageDisplayRotationHelper.getCameraToDisplayRotation());

            // Measure frame time since last successful execution of drawWithCpuImage().
            cpuImageFrameTimeHelper.nextFrame();
        } else {
            cpuImageRenderer.drawWithoutCpuImage();
        }

        // Submit request for the texture from the current frame.
        gpuDownloadFrameBufferIndex =
                textureReader.submitFrame(cpuImageRenderer.getTextureId(), TEXTURE_WIDTH, TEXTURE_HEIGHT);
    }

    public void onLowResolutionRadioButtonClicked(View view) {
        boolean checked = ((RadioButton) view).isChecked();
        if (checked && cpuResolution != ImageResolution.LOW_RESOLUTION) {
            // Display low resolution.
            onCameraConfigChanged(cpuLowResolutionCameraConfig);
            cpuResolution = ImageResolution.LOW_RESOLUTION;
        }
    }

    public void onMediumResolutionRadioButtonClicked(View view) {
        boolean checked = ((RadioButton) view).isChecked();
        if (checked && cpuResolution != ImageResolution.MEDIUM_RESOLUTION) {
            // Display medium resolution.
            onCameraConfigChanged(cpuMediumResolutionCameraConfig);
            cpuResolution = ImageResolution.MEDIUM_RESOLUTION;
        }
    }

    public void onHighResolutionRadioButtonClicked(View view) {
        boolean checked = ((RadioButton) view).isChecked();
        if (checked && cpuResolution != ImageResolution.HIGH_RESOLUTION) {
            // Display high resolution.
            onCameraConfigChanged(cpuHighResolutionCameraConfig);
            cpuResolution = ImageResolution.HIGH_RESOLUTION;
        }
    }

    private void onCVModeChanged(CompoundButton unusedButton, boolean isChecked) {
        cpuImageRenderer.setSplitterPosition(isChecked ? 0.0f : 1.0f);
        isCVModeOn = isChecked;

        // Display the CPU resolution related UI only when CPU image is being displayed.
        boolean show = (cpuImageRenderer.getSplitterPosition() < 0.5f);
        RadioGroup radioGroup = (RadioGroup) findViewById(R.id.radio_camera_configs);
        radioGroup.setVisibility(show ? View.VISIBLE : View.INVISIBLE);
    }

    private void onFocusModeChanged(CompoundButton unusedButton, boolean isChecked) {
        config.setFocusMode(isChecked ? Config.FocusMode.AUTO : Config.FocusMode.FIXED);
        session.configure(config);
    }

    private void onCameraConfigChanged(CameraConfig cameraConfig) {
        // To change the AR camera config - first we pause the AR session, set the desired camera
        // config and then resume the AR session.
        if (session != null) {
            // Block here if the image is still being used.
            synchronized (frameImageInUseLock) {
                session.pause();
                session.setCameraConfig(cameraConfig);
                try {
                    session.resume();
                } catch (CameraNotAvailableException ex) {
                    messageSnackbarHelper.showError(this, "Camera not available. Try restarting the app.");
                    session = null;
                    return;
                }
            }

            // Let the user know that the camera config is set.
            String toastMessage =
                    "Set the camera config with CPU image resolution of "
                            + cameraConfig.getImageSize()
                            + " and fps "
                            + cameraConfig.getFpsRange()
                            + ".";
            Toast toast = Toast.makeText(this, toastMessage, Toast.LENGTH_LONG);
            toast.setGravity(Gravity.BOTTOM, /* xOffset= */ 0, /* yOffset=*/ 250);
            toast.show();
        }
    }

    // Obtains the supported camera configs and build the list of radio button one for each camera
    // config.
    private void obtainCameraConfigs() {
        // First obtain the session handle before getting the list of various camera configs.
        if (session != null) {
            // Create filter here with desired fps filters.
            CameraConfigFilter cameraConfigFilter =
                    new CameraConfigFilter(session)
                            .setTargetFps(
                                    EnumSet.of(
                                            CameraConfig.TargetFps.TARGET_FPS_30, CameraConfig.TargetFps.TARGET_FPS_60));
            List<CameraConfig> cameraConfigs = session.getSupportedCameraConfigs(cameraConfigFilter);
            Log.i(TAG, "Size of supported CameraConfigs list is " + cameraConfigs.size());

            // Determine the highest and lowest CPU resolutions.
            cpuLowResolutionCameraConfig =
                    getCameraConfigWithSelectedResolution(
                            cameraConfigs, /*ImageResolution*/ ImageResolution.LOW_RESOLUTION);
            cpuMediumResolutionCameraConfig =
                    getCameraConfigWithSelectedResolution(
                            cameraConfigs, /*ImageResolution*/ ImageResolution.MEDIUM_RESOLUTION);
            cpuHighResolutionCameraConfig =
                    getCameraConfigWithSelectedResolution(
                            cameraConfigs, /*ImageResolution*/ ImageResolution.HIGH_RESOLUTION);
            // Update the radio buttons with the resolution info.
            updateRadioButtonText(
                    R.id.radio_low_res, cpuLowResolutionCameraConfig, getString(R.string.label_low_res));
            updateRadioButtonText(
                    R.id.radio_medium_res,
                    cpuMediumResolutionCameraConfig,
                    getString(R.string.label_medium_res));
            updateRadioButtonText(
                    R.id.radio_high_res, cpuHighResolutionCameraConfig, getString(R.string.label_high_res));
            cpuResolution = ImageResolution.LOW_RESOLUTION;
        }
    }

    private void updateRadioButtonText(int id, CameraConfig cameraConfig, String prefix) {
        RadioButton radioButton = (RadioButton) findViewById(id);
        Size resolution = cameraConfig.getImageSize();
        radioButton.setText(prefix + " (" + resolution.getWidth() + "x" + resolution.getHeight() + ")");
    }

    /* Get the CameraConfig with selected resolution. */
    private static CameraConfig getCameraConfigWithSelectedResolution(
            List<CameraConfig> cameraConfigs, ImageResolution resolution) {
        // Take the first three camera configs, if camera configs size are larger than 3.
        List<CameraConfig> cameraConfigsByResolution =
                new ArrayList<>(
                        cameraConfigs.subList(0, Math.min(cameraConfigs.size(), 3)));
        Collections.sort(
                cameraConfigsByResolution,
                (CameraConfig p1, CameraConfig p2) ->
                        Integer.compare(p1.getImageSize().getHeight(), p2.getImageSize().getHeight()));
        CameraConfig cameraConfig = cameraConfigsByResolution.get(0);
        switch (resolution) {
            case LOW_RESOLUTION:
                cameraConfig = cameraConfigsByResolution.get(0);
                break;
            case MEDIUM_RESOLUTION:
                // There are some devices that medium resolution is the same as high resolution.
                cameraConfig = cameraConfigsByResolution.get(1);
                break;
            case HIGH_RESOLUTION:
                cameraConfig = cameraConfigsByResolution.get(2);
                break;
        }
        return cameraConfig;
    }

    private String getCameraIntrinsicsText(Frame frame) {
        Camera camera = frame.getCamera();

        boolean forGpuTexture = (cpuImageRenderer.getSplitterPosition() > 0.5f);
        CameraIntrinsics intrinsics =
                forGpuTexture ? camera.getTextureIntrinsics() : camera.getImageIntrinsics();
        String intrinsicsLabel = forGpuTexture ? "Texture" : "Image";
        String imageType = forGpuTexture ? "GPU" : "CPU";

        float[] focalLength = intrinsics.getFocalLength();
        float[] principalPoint = intrinsics.getPrincipalPoint();
        int[] imageSize = intrinsics.getImageDimensions();

        float fovX = (float) (2 * Math.atan2((double) imageSize[0], (double) (2 * focalLength[0])));
        float fovY = (float) (2 * Math.atan2((double) imageSize[1], (double) (2 * focalLength[1])));
        fovX *= RADIANS_TO_DEGREES;
        fovY *= RADIANS_TO_DEGREES;

        return String.format(
                CAMERA_INTRINSICS_TEXT_FORMAT,
                imageType,
                intrinsicsLabel,
                focalLength[0],
                focalLength[1],
                principalPoint[0],
                principalPoint[1],
                imageType,
                imageSize[0],
                imageSize[1],
                fovX,
                fovY,
                renderFrameTimeHelper.getSmoothedFrameTime(),
                renderFrameTimeHelper.getSmoothedFrameRate(),
                cpuImageFrameTimeHelper.getSmoothedFrameTime(),
                cpuImageFrameTimeHelper.getSmoothedFrameRate());
    }

    // 拍摄，在点击按钮时调用
    private void takePhoto() {
        // 创建一个新的线程
        Thread photoThread = new Thread(new Runnable() {
            @Override
            public void run() {
                // 相机的拍摄 use case
                if (imageCapture == null) {
                    return;
                }

                // 创建用于图片输出的文件
                File photoFile = new File(
                        outputDirectory,
                        new SimpleDateFormat(FILENAME_FORMAT, Locale.US)
                                .format(System.currentTimeMillis()) + ".jpg");

                ImageCapture.OutputFileOptions outputOptions = new ImageCapture.OutputFileOptions.Builder(photoFile).build();

                // 拍摄照片后回调
                imageCapture.takePicture(
                        outputOptions, ContextCompat.getMainExecutor(ComputerVisionActivity.this),
                        new ImageCapture.OnImageSavedCallback() {
                            @Override
                            public void onError(ImageCaptureException e) {
                                String msg = "Photo capture failed: " + e.getMessage();
                                Log.e(TAG, msg, e);
                            }

                            @Override
                            public void onImageSaved(ImageCapture.OutputFileResults output) {
                                Uri savedUri = Uri.fromFile(photoFile);

                                photoUris.add(savedUri);
                                rotationMatrices.add(rotationMatrix);
                                translationMatrices.add(translationMatrix);

                                String msg = "Photo capture succeeded: " + savedUri;
                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        Toast.makeText(getApplicationContext(), msg, Toast.LENGTH_SHORT).show();
                                    }
                                });

                                Log.d(TAG, msg);
                            }
                        });
            }
        });

        // 启动线程
        photoThread.start();
    }


    // 预览，在页面创建时开启
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            // 绑定生命周期
            ProcessCameraProvider cameraProvider;
            try {
                cameraProvider = cameraProviderFuture.get();
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
                return;
            }

            Preview preview = new Preview.Builder().build();
            preview.setSurfaceProvider(viewFinder.getSurfaceProvider());

            imageCapture = new ImageCapture.Builder().build();

            CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

            try {
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview, imageCapture);

            } catch (Exception exc) {
                Log.e(TAG, "Use case binding failed", exc);
            }

        }, ContextCompat.getMainExecutor(this));
    }

    private boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(
                getBaseContext(), REQUIRED_PERMISSIONS[0]) == PackageManager.PERMISSION_GRANTED;
    }

    private File getOutputDirectory() {
        File mediaDir = getExternalMediaDirs()[0];
        File appDir = new File(mediaDir, getResources().getString(R.string.app_name));
        if (!appDir.exists()) {
            appDir.mkdirs();
        }
        return appDir;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        calculateRotationMatrix(event);
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            calculateTranslationMatrix(event);
        }
    }

    public void calculateRotationMatrix(SensorEvent event) {
        // 处理传感器数据
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            accelerometerValues = event.values.clone();
        } else if (event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD) {
            magneticFieldValues = event.values.clone();
        }

        // 计算旋转矩阵
        SensorManager.getRotationMatrix(rotationMatrix, null, accelerometerValues, magneticFieldValues);
    }

    public void calculateTranslationMatrix(SensorEvent event) {
        // 获取传感器数据
        accelerometerValues = event.values.clone();
        // 使用卡尔曼滤波器进行预测
        kalmanFilter.predict();
        // 构建测量向量
        RealVector measurement = MatrixUtils.createRealVector(new double[]{accelerometerValues[0], accelerometerValues[1], accelerometerValues[2]});
        // 使用卡尔曼滤波器进行更新
        kalmanFilter.correct(measurement);
        // 获取滤波后的状态
        RealVector filteredState = new ArrayRealVector(kalmanFilter.getStateEstimation());

        float displacementX = (float) filteredState.getEntry(0);
        float displacementY = (float) filteredState.getEntry(1);
        float displacementZ = (float) filteredState.getEntry(2);

        Matrix.setIdentityM(translationMatrix, 0);
        translationMatrix[3] = displacementX;
        translationMatrix[7] = displacementY;
        translationMatrix[11] = displacementZ;
    }

    public void printMatrix(float[] matrix, int rows, int cols) {
        StringBuilder matrixBuilder = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            StringBuilder rowStringBuilder = new StringBuilder();
            for (int j = 0; j < cols; j++) {
                rowStringBuilder.append(matrix[i * cols + j]);
                if (j < cols - 1) {
                    rowStringBuilder.append(" ");
                }
            }
            matrixBuilder.append(rowStringBuilder);
            matrixBuilder.append("\n");
        }
//        runOnUiThread(new Runnable() {
//            @Override
//            public void run() {
//                textView.setText(matrixBuilder.toString());
//            }
//        });

        Log.d(TAG, matrixBuilder.toString());
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // 精度变化时的处理，可以忽略
    }
}
