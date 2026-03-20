import 'dart:io';
import 'package:flutter/material.dart';
import 'detector_screen.dart';

class BatchDetectionResult {
  final String imagePath;
  final String imageName;
  final int aircraftCount;
  final int inferenceMs;
  final List<DetectionBox> boxes;

  BatchDetectionResult({
    required this.imagePath,
    required this.imageName,
    required this.aircraftCount,
    required this.inferenceMs,
    required this.boxes,
  });
}

class BatchResultsScreen extends StatelessWidget {
  final List<BatchDetectionResult> results;

  const BatchResultsScreen({super.key, required this.results});

  @override
  Widget build(BuildContext context) {
    final totalImages = results.length;

    final totalAircraft = results.fold<int>(
      0,
      (sum, r) => sum + r.aircraftCount,
    );

    final avgInference = results.isEmpty
        ? 0
        : results.fold<int>(0, (sum, r) => sum + r.inferenceMs) ~/
            results.length;

    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FA),
      appBar: AppBar(
        title: const Text("Batch Results"),
        backgroundColor: Colors.white,
        elevation: 0,
      ),
      body: results.isEmpty
          ? const Center(
              child: Text(
                "No batch results available",
                style: TextStyle(fontSize: 16),
              ),
            )
          : Column(
              children: [
                // 🔹 SUMMARY CARD
                Container(
                  margin: const EdgeInsets.all(16),
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
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        "Batch Summary",
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text("Images processed: $totalImages"),
                      Text("Total aircraft detected: $totalAircraft"),
                      Text("Average inference: $avgInference ms"),
                    ],
                  ),
                ),

                // 🔹 LIST
                Expanded(
                  child: ListView.builder(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    itemCount: results.length,
                    itemBuilder: (context, index) {
                      final item = results[index];

                      return InkWell(
                        borderRadius: BorderRadius.circular(16),
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => DetectorScreen(
                                initialImagePath: item.imagePath,
                                initialBoxes: item.boxes,
                              ),
                            ),
                          );
                        },
                        child: Container(
                          margin: const EdgeInsets.only(bottom: 14),
                          padding: const EdgeInsets.all(14),
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
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              ClipRRect(
                                borderRadius: BorderRadius.circular(12),
                                child: Image.file(
                                  File(item.imagePath),
                                  width: 85,
                                  height: 85,
                                  fit: BoxFit.cover,
                                  errorBuilder: (_, __, ___) {
                                    return Container(
                                      width: 85,
                                      height: 85,
                                      color: Colors.grey.shade300,
                                      child: const Icon(Icons.image, size: 36),
                                    );
                                  },
                                ),
                              ),
                              const SizedBox(width: 14),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      item.imageName,
                                      style: const TextStyle(
                                        fontSize: 17,
                                        fontWeight: FontWeight.w700,
                                      ),
                                    ),
                                    const SizedBox(height: 8),
                                    Text(
                                      "✈️ ${item.aircraftCount} aircraft",
                                      style: const TextStyle(fontSize: 14),
                                    ),
                                    const SizedBox(height: 4),
                                    Text(
                                      "⚡ ${item.inferenceMs} ms",
                                      style: const TextStyle(fontSize: 14),
                                    ),
                                  ],
                                ),
                              ),
                              const Icon(
                                Icons.arrow_forward_ios,
                                size: 16,
                                color: Colors.grey,
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
    );
  }
}