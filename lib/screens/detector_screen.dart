import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'batch_results_screen.dart';

class DetectionBox {
  final double left;
  final double top;
  final double right;
  final double bottom;
  final double confidence;

  const DetectionBox({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
    required this.confidence,
  });

  Map<String, dynamic> toJson() {
    return {
      'left': left,
      'top': top,
      'right': right,
      'bottom': bottom,
      'confidence': confidence,
    };
  }

  factory DetectionBox.fromJson(Map<String, dynamic> json) {
    return DetectionBox(
      left: (json['left'] as num).toDouble(),
      top: (json['top'] as num).toDouble(),
      right: (json['right'] as num).toDouble(),
      bottom: (json['bottom'] as num).toDouble(),
      confidence: (json['confidence'] as num).toDouble(),
    );
  }
}

class DetectorScreen extends StatefulWidget {
  final String? initialImagePath;
  final List<DetectionBox>? initialBoxes;

  const DetectorScreen({
    super.key,
    this.initialImagePath,
    this.initialBoxes,
  });

  @override
  State<DetectorScreen> createState() => _DetectorScreenState();
}

class ImageTile {
  final img.Image image;
  final int x;
  final int y;

  ImageTile({
    required this.image,
    required this.x,
    required this.y,
  });
}

class _DetectorScreenState extends State<DetectorScreen> {
  File? selectedImage;
  String status = "Loading YOLO model...";
  String detectionText = "No detections yet.";

  OrtSession? session;
  List<DetectionBox> detections = [];
  List<DetectionBox> rawDetections = [];
  bool hasCachedDetections = false;

  int originalImageWidth = 0;
  int originalImageHeight = 0;
  int inferenceMs = 0;

  static const int modelSize = 640;
  double confidenceThreshold = 0.40;
  double nmsThreshold = 0.45;
  bool useTiling = false;
  bool isDetecting = false;

  static const int tileSize = 640;
  static const int tileOverlap = 100;

  final TransformationController _transformController =
      TransformationController();

  Future<void> loadInitialImageDimensions(File file) async {
    final bytes = await file.readAsBytes();
    final decoded = img.decodeImage(bytes);

    if (decoded == null) return;

    setState(() {
      originalImageWidth = decoded.width;
      originalImageHeight = decoded.height;
    });
  }

  List<ImageTile> splitIntoTiles(img.Image image) {
    final tiles = <ImageTile>[];
    final step = tileSize - tileOverlap;

    for (int y = 0; y < image.height; y += step) {
      for (int x = 0; x < image.width; x += step) {
        final w = (x + tileSize > image.width) ? image.width - x : tileSize;
        final h = (y + tileSize > image.height) ? image.height - y : tileSize;

        final tile = img.copyCrop(
          image,
          x: x,
          y: y,
          width: w,
          height: h,
        );

        tiles.add(
          ImageTile(
            image: tile,
            x: x,
            y: y,
          ),
        );
      }
    }

    return tiles;
  }

  bool shouldSkipTile(img.Image tile) {
    return false;
  }

  Future<void> runDetectionTiled(File file) async {
    final imageBytes = await file.readAsBytes();
    final decoded = img.decodeImage(imageBytes);

    if (decoded == null) {
      setState(() {
        detectionText = "Could not decode image for tiling.";
      });
      return;
    }

    final tiles = splitIntoTiles(decoded);
    final allBoxes = <DetectionBox>[];

    int skippedTiles = 0;

    for (final tile in tiles) {
      if (shouldSkipTile(tile.image)) {
        skippedTiles++;
        continue;
      }

      final tileBoxes = await runDetectionOnDecodedImage(tile.image);

      for (final box in tileBoxes) {
        allBoxes.add(
          DetectionBox(
            left: box.left + tile.x,
            top: box.top + tile.y,
            right: box.right + tile.x,
            bottom: box.bottom + tile.y,
            confidence: box.confidence,
          ),
        );
      }
    }

    final filteredBoxes = applyNms(allBoxes);

    setState(() {
      selectedImage = file;
      originalImageWidth = decoded.width;
      originalImageHeight = decoded.height;
      detections = filteredBoxes;
      inferenceMs = 0;
      detectionText =
          "Tiled detection found ${filteredBoxes.length} boxes from ${tiles.length} tiles ($skippedTiles skipped).";
    });

    _resetZoom();
  }

  Future<List<DetectionBox>> runDetectionOnDecodedImage(img.Image image) async {
    if (session == null) return [];

    final resized = img.copyResize(
      image,
      width: modelSize,
      height: modelSize,
    );

    final input = Float32List(modelSize * modelSize * 3);

    for (int y = 0; y < modelSize; y++) {
      for (int x = 0; x < modelSize; x++) {
        final pixel = resized.getPixel(x, y);
        final pixelIndex = y * modelSize + x;

        input[pixelIndex] = pixel.r / 255.0;
        input[modelSize * modelSize + pixelIndex] = pixel.g / 255.0;
        input[2 * modelSize * modelSize + pixelIndex] = pixel.b / 255.0;
      }
    }

    OrtValueTensor? inputTensor;
    OrtRunOptions? runOptions;
    List<OrtValue?>? outputs;

    try {
      inputTensor = OrtValueTensor.createTensorWithDataList(
        input,
        [1, 3, modelSize, modelSize],
      );

      runOptions = OrtRunOptions();
      outputs = await session!.runAsync(runOptions, {'images': inputTensor});

      final rawOutput = outputs?.first?.value;
      if (rawOutput == null) return [];

      final outputData = rawOutput as List;
      final channels = outputData.first as List;

      final xs = List<num>.from(channels[0] as List);
      final ys = List<num>.from(channels[1] as List);
      final ws = List<num>.from(channels[2] as List);
      final hs = List<num>.from(channels[3] as List);
      final confs = List<num>.from(channels[4] as List);

      final candidates = <DetectionBox>[];

      for (int i = 0; i < confs.length; i++) {
        final confidence = confs[i].toDouble();
        if (confidence < confidenceThreshold) continue;

        final cx = xs[i].toDouble();
        final cy = ys[i].toDouble();
        final w = ws[i].toDouble();
        final h = hs[i].toDouble();

        double left = cx - w / 2.0;
        double top = cy - h / 2.0;
        double right = cx + w / 2.0;
        double bottom = cy + h / 2.0;

        left = left.clamp(0, modelSize.toDouble());
        top = top.clamp(0, modelSize.toDouble());
        right = right.clamp(0, modelSize.toDouble());
        bottom = bottom.clamp(0, modelSize.toDouble());

        final scaleX = image.width / modelSize;
        final scaleY = image.height / modelSize;

        candidates.add(
          DetectionBox(
            left: left * scaleX,
            top: top * scaleY,
            right: right * scaleX,
            bottom: bottom * scaleY,
            confidence: confidence,
          ),
        );
      }

      return candidates;
    } finally {
      inputTensor?.release();
      runOptions?.release();

      if (outputs != null) {
        for (final out in outputs) {
          out?.release();
        }
      }
    }
  }

  double computeIoU(DetectionBox a, DetectionBox b) {
    final left = math.max(a.left, b.left);
    final top = math.max(a.top, b.top);
    final right = math.min(a.right, b.right);
    final bottom = math.min(a.bottom, b.bottom);

    final intersectionWidth = math.max(0.0, right - left);
    final intersectionHeight = math.max(0.0, bottom - top);
    final intersectionArea = intersectionWidth * intersectionHeight;

    final areaA = (a.right - a.left) * (a.bottom - a.top);
    final areaB = (b.right - b.left) * (b.bottom - b.top);

    final unionArea = areaA + areaB - intersectionArea;
    if (unionArea <= 0) return 0.0;

    return intersectionArea / unionArea;
  }

  List<DetectionBox> applyNms(List<DetectionBox> boxes) {
    final sorted = List<DetectionBox>.from(boxes)
      ..sort((a, b) => b.confidence.compareTo(a.confidence));

    final selected = <DetectionBox>[];

    while (sorted.isNotEmpty) {
      final current = sorted.removeAt(0);
      selected.add(current);

      sorted.removeWhere((box) => computeIoU(current, box) > nmsThreshold);
    }

    return selected;
  }

  void _updateDetections() {
    if (hasCachedDetections) {
      setState(() {
        detections = applyNms(
          rawDetections
              .where((d) => d.confidence >= confidenceThreshold)
              .toList(),
        );
        detectionText = "Aircraft detected: ${detections.length}";
      });
    }
  }

  void _resetZoom() {
    _transformController.value = Matrix4.identity();
  }

  void _zoom(double scaleFactor) {
    final currentScale = _transformController.value.getMaxScaleOnAxis();
    final newScale = (currentScale * scaleFactor).clamp(1.0, 8.0);
    _transformController.value = Matrix4.identity()..scale(newScale);
  }

  Widget _card({required Widget child}) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: const [
          BoxShadow(
            color: Colors.black12,
            blurRadius: 6,
          ),
        ],
      ),
      child: child,
    );
  }

  @override
  void initState() {
    super.initState();
    loadModel();

    if (widget.initialImagePath != null) {
      final file = File(widget.initialImagePath!);

      if (file.existsSync()) {
        selectedImage = file;
        loadInitialImageDimensions(file);

        if (widget.initialBoxes != null && widget.initialBoxes!.isNotEmpty) {
          setState(() {
            detections = widget.initialBoxes!;
            detectionText = "Aircraft detected: ${widget.initialBoxes!.length}";
            inferenceMs = 0;
          });
        } else {
          setState(() {
            detections = [];
            detectionText = "Waiting for model...";
            inferenceMs = 0;
          });
        }
      }
    }
  }

  Future<void> loadModel() async {
    try {
      final modelData = await rootBundle.load("assets/yolo11l-best.onnx");

      final sessionOptions = OrtSessionOptions();
      session = OrtSession.fromBuffer(
        modelData.buffer.asUint8List(),
        sessionOptions,
      );

      setState(() {
        status = "YOLO model loaded successfully!";
      });

      if (widget.initialImagePath != null &&
          selectedImage != null &&
          (widget.initialBoxes == null || widget.initialBoxes!.isEmpty)) {
        await runDetection(selectedImage!);
      }
    } catch (e) {
      setState(() {
        status = "Model failed to load: $e";
      });
    }
  }

  Future<void> pickImage() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.image);
    if (result == null || result.files.single.path == null) return;

    final file = File(result.files.single.path!);

    setState(() {
      selectedImage = file;
      detections = [];
      detectionText = "Running detection...";
      inferenceMs = 0;
    });

    _resetZoom();
    await runDetection(file);
  }

  Future<void> pickMultipleImages() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      allowMultiple: true,
    );

    if (result == null || result.files.isEmpty) return;

    final List<BatchDetectionResult> batchResults = [];

    for (final picked in result.files) {
      if (picked.path == null) continue;

      final file = File(picked.path!);

      setState(() {
        selectedImage = file;
        detections = [];
        detectionText = "Running batch detection...";
        inferenceMs = 0;
      });

      _resetZoom();
      await runDetection(file);

      batchResults.add(
        BatchDetectionResult(
          imagePath: file.path,
          imageName: file.path.split(Platform.pathSeparator).last,
          aircraftCount: detections.length,
          inferenceMs: inferenceMs,
          boxes: List<DetectionBox>.from(detections),
        ),
      );
    }

    if (!mounted) return;

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => BatchResultsScreen(results: batchResults),
      ),
    );
  }

  Future<void> saveDetectionHistory({
    required String imagePath,
    required int aircraftCount,
    required int inferenceMs,
    required List<Map<String, dynamic>> boxes,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    final existing = prefs.getStringList('detection_history') ?? [];

    final item = {
      'imagePath': imagePath,
      'imageName': imagePath.split(Platform.pathSeparator).last,
      'aircraftCount': aircraftCount,
      'inferenceMs': inferenceMs,
      'timestamp': DateTime.now().toString(),
      'boxes': boxes,
    };

    existing.add(jsonEncode(item));
    await prefs.setStringList('detection_history', existing);
  }

  Future<void> runDetection(File file) async {
    if (isDetecting) return;

    isDetecting = true;

    try {
      if (useTiling) {
        await runDetectionTiled(file);
        return;
      }

      if (session == null) {
        setState(() {
          detectionText = "Model is not loaded.";
        });
        return;
      }

      OrtValueTensor? inputTensor;
      OrtRunOptions? runOptions;
      List<OrtValue?>? outputs;

      try {
        final stopwatch = Stopwatch()..start();

        final imageBytes = await file.readAsBytes();
        final decoded = img.decodeImage(imageBytes);

        if (decoded == null) {
          setState(() {
            detectionText = "Could not decode image.";
          });
          return;
        }

        originalImageWidth = decoded.width;
        originalImageHeight = decoded.height;

        final resized = img.copyResize(
          decoded,
          width: modelSize,
          height: modelSize,
        );

        final input = Float32List(1 * 3 * modelSize * modelSize);
        int pixelIndex = 0;

        for (int y = 0; y < modelSize; y++) {
          for (int x = 0; x < modelSize; x++) {
            final pixel = resized.getPixel(x, y);

            input[pixelIndex] = pixel.r / 255.0;
            input[modelSize * modelSize + pixelIndex] = pixel.g / 255.0;
            input[2 * modelSize * modelSize + pixelIndex] = pixel.b / 255.0;

            pixelIndex++;
          }
        }

        inputTensor = OrtValueTensor.createTensorWithDataList(
          input,
          [1, 3, modelSize, modelSize],
        );

        runOptions = OrtRunOptions();
        outputs = session!.run(
          runOptions,
          {'images': inputTensor},
        );

        final raw = outputs[0]?.value;
        if (raw == null) {
          setState(() {
            detectionText = "No model output returned.";
          });
          return;
        }

        final rawList = raw as List;
        final channels = rawList[0] as List;
        if (channels.length < 5) {
          setState(() {
            detectionText = "Unexpected output shape.";
          });
          return;
        }

        final xs = List<num>.from(channels[0] as List);
        final ys = List<num>.from(channels[1] as List);
        final ws = List<num>.from(channels[2] as List);
        final hs = List<num>.from(channels[3] as List);
        final confs = List<num>.from(channels[4] as List);

        final candidates = <DetectionBox>[];

        for (int i = 0; i < confs.length; i++) {
          final confidence = confs[i].toDouble();
          if (confidence < confidenceThreshold) continue;

          final cx = xs[i].toDouble();
          final cy = ys[i].toDouble();
          final w = ws[i].toDouble();
          final h = hs[i].toDouble();

          if (w > 300 || h > 300) continue;

          double left = cx - w / 2.0;
          double top = cy - h / 2.0;
          double right = cx + w / 2.0;
          double bottom = cy + h / 2.0;

          left = left.clamp(0.0, modelSize.toDouble());
          top = top.clamp(0.0, modelSize.toDouble());
          right = right.clamp(0.0, modelSize.toDouble());
          bottom = bottom.clamp(0.0, modelSize.toDouble());

          if (right <= left || bottom <= top) continue;

          final scaleX = originalImageWidth / modelSize;
          final scaleY = originalImageHeight / modelSize;

          candidates.add(
            DetectionBox(
              left: left * scaleX,
              top: top * scaleY,
              right: right * scaleX,
              bottom: bottom * scaleY,
              confidence: confidence,
            ),
          );
        }

        stopwatch.stop();

        rawDetections = candidates;
        hasCachedDetections = true;

        setState(() {
          detections = applyNms(
            rawDetections
                .where((d) => d.confidence >= confidenceThreshold)
                .toList(),
          );
          inferenceMs = stopwatch.elapsedMilliseconds;
          detectionText = "Aircraft detected: ${detections.length}";
        });

        await saveDetectionHistory(
          imagePath: file.path,
          aircraftCount: detections.length,
          inferenceMs: stopwatch.elapsedMilliseconds,
          boxes: detections.map((b) => b.toJson()).toList(),
        );
      } catch (e) {
        setState(() {
          detectionText = "Detection error: $e";
        });
      } finally {
        inputTensor?.release();
        runOptions?.release();
        if (outputs != null) {
          for (final out in outputs) {
            out?.release();
          }
        }
      }
    } finally {
      isDetecting = false;
    }
  }

  double iou(DetectionBox a, DetectionBox b) {
    final interLeft = math.max(a.left, b.left);
    final interTop = math.max(a.top, b.top);
    final interRight = math.min(a.right, b.right);
    final interBottom = math.min(a.bottom, b.bottom);

    final interWidth = math.max(0.0, interRight - interLeft);
    final interHeight = math.max(0.0, interBottom - interTop);
    final interArea = interWidth * interHeight;

    final areaA = (a.right - a.left) * (a.bottom - a.top);
    final areaB = (b.right - b.left) * (b.bottom - b.top);
    final union = areaA + areaB - interArea;

    if (union <= 0) return 0.0;
    return interArea / union;
  }

  @override
  void dispose() {
    _transformController.dispose();
    session?.release();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final metricsText =
        "Threshold: $confidenceThreshold   •   NMS: $nmsThreshold   •   Inference: $inferenceMs ms";

    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FA),
      appBar: AppBar(
        title: const Text("Aircraft Detector"),
        backgroundColor: Colors.white,
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _card(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  status,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  detectionText,
                  style: const TextStyle(fontSize: 15),
                ),
                const SizedBox(height: 6),
                Text(
                  metricsText,
                  style: TextStyle(
                    fontSize: 13,
                    color: Colors.grey.shade700,
                  ),
                ),
              ],
            ),
          ),
          _card(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  "Detection Controls",
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  "Confidence Threshold: ${confidenceThreshold.toStringAsFixed(2)}",
                ),
                Slider(
                  value: confidenceThreshold,
                  min: 0.1,
                  max: 0.9,
                  divisions: 16,
                  label: confidenceThreshold.toStringAsFixed(2),
                  onChanged: (value) {
                    setState(() {
                      confidenceThreshold = value;
                    });
                  },
                  onChangeEnd: (_) => _updateDetections(),
                ),
                const SizedBox(height: 8),
                Text(
                  "NMS Threshold: ${nmsThreshold.toStringAsFixed(2)}",
                ),
                Slider(
                  value: nmsThreshold,
                  min: 0.1,
                  max: 0.9,
                  divisions: 16,
                  label: nmsThreshold.toStringAsFixed(2),
                  onChanged: (value) {
                    setState(() {
                      nmsThreshold = value;
                    });
                  },
                  onChangeEnd: (_) => _updateDetections(),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    const Text("Use Tiling"),
                    const SizedBox(width: 10),
                    Switch(
                      value: useTiling,
                      onChanged: (value) {
                        setState(() {
                          useTiling = value;
                        });
                      },
                    ),
                  ],
                ),
              ],
            ),
          ),
          _card(
            child: Column(
              children: [
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: pickImage,
                    icon: const Icon(Icons.image_search),
                    label: const Text("Select Satellite Image"),
                  ),
                ),
                const SizedBox(height: 10),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: pickMultipleImages,
                    icon: const Icon(Icons.collections),
                    label: const Text("Detect Multiple Images"),
                  ),
                ),
              ],
            ),
          ),
          _card(
            child: SizedBox(
              height: 380,
              child: selectedImage == null
                  ? const Center(
                      child: Text(
                        "No Image Selected",
                        style: TextStyle(fontSize: 16),
                      ),
                    )
                  : ClipRect(
                      child: Stack(
                        children: [
                          Positioned.fill(
                            child: InteractiveViewer(
                              transformationController: _transformController,
                              panEnabled: true,
                              scaleEnabled: true,
                              minScale: 1.0,
                              maxScale: 8.0,
                              boundaryMargin: const EdgeInsets.all(20),
                              child: SizedBox.expand(
                                child: Stack(
                                  children: [
                                    Positioned.fill(
                                      child: FittedBox(
                                        fit: BoxFit.contain,
                                        child: SizedBox(
                                          width:
                                              originalImageWidth.toDouble(),
                                          height:
                                              originalImageHeight.toDouble(),
                                          child: Image.file(selectedImage!),
                                        ),
                                      ),
                                    ),
                                    Positioned.fill(
                                      child: CustomPaint(
                                        painter: DetectionPainter(
                                          detections: detections,
                                          imageWidth: originalImageWidth,
                                          imageHeight: originalImageHeight,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                          Positioned(
                            right: 12,
                            bottom: 12,
                            child: Column(
                              children: [
                                FloatingActionButton.small(
                                  heroTag: "zoom_in",
                                  onPressed: () => _zoom(1.2),
                                  child: const Icon(Icons.add),
                                ),
                                const SizedBox(height: 8),
                                FloatingActionButton.small(
                                  heroTag: "zoom_out",
                                  onPressed: () => _zoom(0.8),
                                  child: const Icon(Icons.remove),
                                ),
                                const SizedBox(height: 8),
                                FloatingActionButton.small(
                                  heroTag: "zoom_reset",
                                  onPressed: _resetZoom,
                                  child: const Icon(Icons.refresh),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
            ),
          ),
        ],
      ),
    );
  }
}

class DetectionPainter extends CustomPainter {
  final List<DetectionBox> detections;
  final int imageWidth;
  final int imageHeight;

  DetectionPainter({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageWidth == 0 || imageHeight == 0 || detections.isEmpty) return;

    final imageAspect = imageWidth / imageHeight;
    final canvasAspect = size.width / size.height;

    double drawWidth;
    double drawHeight;
    double offsetX = 0;
    double offsetY = 0;

    if (canvasAspect > imageAspect) {
      drawHeight = size.height;
      drawWidth = drawHeight * imageAspect;
      offsetX = (size.width - drawWidth) / 2;
    } else {
      drawWidth = size.width;
      drawHeight = drawWidth / imageAspect;
      offsetY = (size.height - drawHeight) / 2;
    }

    final scaleX = drawWidth / imageWidth;
    final scaleY = drawHeight / imageHeight;

    Color getBoxColor(double confidence) {
      if (confidence >= 0.90) return Colors.green;
      if (confidence >= 0.75) return Colors.yellow;
      return Colors.deepOrange;
    }

    for (final box in detections) {
      final boxColor = getBoxColor(box.confidence);

      final rectPaint = Paint()
        ..color = boxColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;

      final labelBgPaint = Paint()..color = boxColor;

      final rect = Rect.fromLTRB(
        offsetX + box.left * scaleX,
        offsetY + box.top * scaleY,
        offsetX + box.right * scaleX,
        offsetY + box.bottom * scaleY,
      );

      canvas.drawRect(rect, rectPaint);

      final label = box.confidence.toStringAsFixed(2);

      final outlinePainter = TextPainter(
        text: TextSpan(
          text: label,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.bold,
            foreground: Paint()
              ..style = PaintingStyle.stroke
              ..strokeWidth = 2
              ..color = Colors.white,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final fillPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.bold,
            color: Colors.black,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final bgRect = Rect.fromLTWH(
        rect.left,
        math.max(0, rect.top - 22),
        fillPainter.width + 6,
        fillPainter.height + 4,
      );

      canvas.drawRect(bgRect, labelBgPaint);
      outlinePainter.paint(canvas, Offset(bgRect.left + 3, bgRect.top + 2));
      fillPainter.paint(canvas, Offset(bgRect.left + 3, bgRect.top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.imageWidth != imageWidth ||
        oldDelegate.imageHeight != imageHeight;
  }
}