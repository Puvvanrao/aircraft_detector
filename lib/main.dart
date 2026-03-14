import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

void main() {
  runApp(const AircraftDetectorApp());
}

class AircraftDetectorApp extends StatelessWidget {
  const AircraftDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: DetectorHomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

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
}

class DetectorHomePage extends StatefulWidget {
  const DetectorHomePage({super.key});

  @override
  State<DetectorHomePage> createState() => _DetectorHomePageState();
}

class _DetectorHomePageState extends State<DetectorHomePage> {
  File? selectedImage;
  String status = "Loading YOLO model...";
  String detectionText = "No detections yet.";

  OrtSession? session;
  List<DetectionBox> detections = [];

  int originalImageWidth = 0;
  int originalImageHeight = 0;
  int inferenceMs = 0;

  static const int modelSize = 640;
  static const double confidenceThreshold = 0.30;
  static const double nmsThreshold = 0.55;

  @override
  void initState() {
    super.initState();
    loadModel();
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

    await runDetection(file);
  }

  Future<void> runDetection(File file) async {
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

      final resized = img.copyResize(decoded, width: modelSize, height: modelSize);

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

        candidates.add(
          DetectionBox(
            left: left,
            top: top,
            right: right,
            bottom: bottom,
            confidence: confidence,
          ),
        );
      }

      final filtered = applyNms(candidates, nmsThreshold);

      stopwatch.stop();

      setState(() {
        detections = filtered;
        inferenceMs = stopwatch.elapsedMilliseconds;
        detectionText = "Aircraft detected: ${filtered.length}";
      });
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
  }

  List<DetectionBox> applyNms(List<DetectionBox> boxes, double iouThreshold) {
    if (boxes.isEmpty) return [];

    final sorted = [...boxes]
      ..sort((a, b) => b.confidence.compareTo(a.confidence));

    final selected = <DetectionBox>[];

    while (sorted.isNotEmpty) {
      final current = sorted.removeAt(0);
      selected.add(current);
      sorted.removeWhere((box) => iou(current, box) > iouThreshold);
    }

    return selected;
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
    session?.release();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final metricsText =
        "Threshold: $confidenceThreshold   •   NMS: $nmsThreshold   •   Inference: ${inferenceMs} ms";

    return Scaffold(
      appBar: AppBar(
        title: const Text("Aircraft Detector"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Text(
              status,
              style: const TextStyle(fontSize: 18),
            ),
            const SizedBox(height: 10),
            Text(
              detectionText,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 6),
            Text(
              metricsText,
              style: const TextStyle(fontSize: 14),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: pickImage,
              child: const Text("Select Satellite Image"),
            ),
            const SizedBox(height: 20),
            Expanded(
              child: Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                ),
                child: selectedImage == null
                    ? const Center(child: Text("No Image Selected"))
                    : InteractiveViewer(
                        minScale: 1.0,
                        maxScale: 8.0,
                        child: LayoutBuilder(
                          builder: (context, constraints) {
                            return Stack(
                              children: [
                                Positioned.fill(
                                  child: Image.file(
                                    selectedImage!,
                                    fit: BoxFit.contain,
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
                            );
                          },
                        ),
                      ),
              ),
            ),
          ],
        ),
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

    const modelSize = 640.0;

    final imageAspect = imageWidth / imageHeight;
    final boxAspect = size.width / size.height;

    double drawWidth;
    double drawHeight;
    double offsetX = 0;
    double offsetY = 0;

    if (imageAspect > boxAspect) {
      drawWidth = size.width;
      drawHeight = size.width / imageAspect;
      offsetY = (size.height - drawHeight) / 2;
    } else {
      drawHeight = size.height;
      drawWidth = size.height * imageAspect;
      offsetX = (size.width - drawWidth) / 2;
    }

    final scaleX = drawWidth / modelSize;
    final scaleY = drawHeight / modelSize;

    final rectPaint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final labelBgPaint = Paint()
      ..color = Colors.red;

    const textStyle = TextStyle(
      color: Colors.white,
      fontSize: 11,
      fontWeight: FontWeight.bold,
    );

    for (final box in detections) {
      final rect = Rect.fromLTRB(
        offsetX + box.left * scaleX,
        offsetY + box.top * scaleY,
        offsetX + box.right * scaleX,
        offsetY + box.bottom * scaleY,
      );

      canvas.drawRect(rect, rectPaint);

      final label = "Aircraft ${box.confidence.toStringAsFixed(2)}";
      final textSpan = TextSpan(text: label, style: textStyle);
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();

      final bgRect = Rect.fromLTWH(
        rect.left,
        math.max(0, rect.top - 16),
        textPainter.width + 6,
        textPainter.height + 4,
      );

      canvas.drawRect(bgRect, labelBgPaint);
      textPainter.paint(canvas, Offset(bgRect.left + 3, bgRect.top + 2));
    }
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.imageWidth != imageWidth ||
        oldDelegate.imageHeight != imageHeight;
  }
}