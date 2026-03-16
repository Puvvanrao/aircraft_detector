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
        : results.fold<int>(0, (sum, r) => sum + r.inferenceMs) ~/ results.length;

    return Scaffold(
      appBar: AppBar(
        title: const Text("Batch Results"),
      ),
      body: results.isEmpty
          ? const Center(
              child: Text("No batch results available"),
            )
          
          : Column(
              children: [

                Padding(
                  padding: const EdgeInsets.all(12),
                  child: Card(
                    elevation: 2,
                    child: Padding(
                      padding: const EdgeInsets.all(12),
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
                          const SizedBox(height: 8),
                          Text("Images processed: $totalImages"),
                          Text("Total aircraft detected: $totalAircraft"),
                          Text("Average inference time: $avgInference ms"),
                        ],
                      ),
                    ),
                  ),
                ),

                Expanded(
                  child: ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: results.length,
              itemBuilder: (context, index) {
                final item = results[index];

                return InkWell(
                  borderRadius: BorderRadius.circular(12),
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
                  child: Card(
                  margin: const EdgeInsets.only(bottom: 14),
                  elevation: 2,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: Image.file(
                            File(item.imagePath),
                            width: 90,
                            height: 90,
                            fit: BoxFit.cover,
                            errorBuilder: (_, __, ___) {
                              return Container(
                                width: 90,
                                height: 90,
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
                                  fontSize: 18,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text("Aircraft detected: ${item.aircraftCount}"),
                              const SizedBox(height: 4),
                              Text("Inference time: ${item.inferenceMs} ms"),
                            ],
                          ),
                        ),
                      ],
                    ),
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