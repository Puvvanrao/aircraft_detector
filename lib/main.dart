import 'package:flutter/material.dart';
import 'screens/login_screen.dart';

void main() {
  runApp(const AircraftDetectorApp());
}

class AircraftDetectorApp extends StatelessWidget {
  const AircraftDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: LoginScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}