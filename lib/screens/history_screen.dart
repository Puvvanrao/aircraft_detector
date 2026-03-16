import 'dart:convert';
import 'dart:io';
import 'detector_screen.dart';

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:shared_preferences/shared_preferences.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<Map<String, dynamic>> historyItems = [];

  @override
  void initState() {
    super.initState();
    loadHistory();
  }

  Future<void> loadHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getStringList('detection_history') ?? [];

    final decoded = saved
        .map((item) => jsonDecode(item) as Map<String, dynamic>)
        .toList();

    setState(() {
      historyItems = decoded.reversed.toList();
    });
  }

  Future<void> clearHistory() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('detection_history');

    setState(() {
      historyItems = [];
    });
  }

  String formatTimestamp(String raw) {
    try {
      final dt = DateTime.parse(raw);
      return DateFormat('MMM d, yyyy • h:mm a').format(dt);
    } catch (_) {
      return raw;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Detection History"),
        actions: [
          IconButton(
            onPressed: historyItems.isEmpty ? null : clearHistory,
            icon: const Icon(Icons.delete),
            tooltip: "Clear history",
          ),
        ],
      ),
      body: historyItems.isEmpty
          ? const Center(
              child: Text(
                "No detection history yet",
                style: TextStyle(fontSize: 16),
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: historyItems.length,
              itemBuilder: (context, index) {
                final item = historyItems[index];
                final imagePath = item['imagePath'] ?? '';
                final imageName = item['imageName'] ?? 'Unknown image';
                final aircraftCount = item['aircraftCount'] ?? 0;
                final inferenceMs = item['inferenceMs'] ?? 0;
                final timestamp = item['timestamp'] ?? '';

                final imageFile = File(imagePath);
                final imageExists = imagePath.isNotEmpty && imageFile.existsSync();

                return InkWell(
                  borderRadius: BorderRadius.circular(16),
                  onTap: () {
                    if (imageExists) {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => DetectorScreen(initialImagePath: imagePath),
                        ),
                      );
                    }
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
                          child: imageExists
                              ? Image.file(
                                  imageFile,
                                  width: 90,
                                  height: 90,
                                  fit: BoxFit.cover,
                                )
                              : Container(
                                  width: 90,
                                  height: 90,
                                  color: Colors.grey.shade300,
                                  child: const Icon(Icons.image, size: 36),
                                ),
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                imageName,
                                style: const TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                "Aircraft detected: $aircraftCount",
                                style: const TextStyle(fontSize: 15),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                "Inference time: $inferenceMs ms",
                                style: const TextStyle(fontSize: 15),
                              ),
                              const SizedBox(height: 6),
                              Text(
                                formatTimestamp(timestamp),
                                style: TextStyle(
                                  fontSize: 13,
                                  color: Colors.grey.shade700,
                                ),
                              ),
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
    );
  }
}