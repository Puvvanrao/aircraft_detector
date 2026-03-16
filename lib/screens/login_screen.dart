import 'package:flutter/material.dart';
import 'package:local_auth/local_auth.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'home_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final LocalAuthentication auth = LocalAuthentication();

  final TextEditingController usernameController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();

  bool biometricEnabled = false;
  bool isReturningUser = false;
  String status = "Login to continue";

  @override
  void initState() {
    super.initState();
    loadLoginState();
  }

  Future<void> loadLoginState() async {
    final prefs = await SharedPreferences.getInstance();

    final savedLoggedIn = prefs.getBool('logged_in') ?? false;
    final savedBiometric = prefs.getBool('biometric_enabled') ?? false;

    setState(() {
      isReturningUser = savedLoggedIn;
      biometricEnabled = savedBiometric;
      status = savedLoggedIn
          ? "Welcome back"
          : "Login with username and password";
    });

    if (savedLoggedIn && savedBiometric) {
      await authenticateWithBiometrics();
    }
  }

  Future<void> loginWithPassword() async {
    final username = usernameController.text.trim();
    final password = passwordController.text.trim();

    // Demo credentials for now
    if (username == 'admin' && password == 'aircraft123') {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool('logged_in', true);

      if (!mounted) return;

      final enableBiometric = await showDialog<bool>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text("Enable Quick Unlock"),
          content: const Text(
            "Would you like to enable biometric or PIN quick unlock for next time?",
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text("Skip"),
            ),
            ElevatedButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text("Enable"),
            ),
          ],
        ),
      );

      await prefs.setBool('biometric_enabled', enableBiometric ?? false);

      if (!mounted) return;

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const HomeScreen()),
      );
    } else {
      setState(() {
        status = "Invalid username or password";
      });
    }
  }

  Future<void> authenticateWithBiometrics() async {
    try {
      final didAuthenticate = await auth.authenticate(
        localizedReason: 'Unlock Aircraft Detector',
        options: const AuthenticationOptions(
          biometricOnly: false,
          stickyAuth: true,
        ),
      );

      if (!mounted) return;

      if (didAuthenticate) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => const HomeScreen()),
        );
      } else {
        setState(() {
          status = "Authentication failed. Use password instead.";
        });
      }
    } catch (e) {
      setState(() {
        status = "Biometric unavailable. Use password instead.";
      });
    }
  }

  @override
  void dispose() {
    usernameController.dispose();
    passwordController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final showBiometricButton = isReturningUser && biometricEnabled;

    return Scaffold(
      appBar: AppBar(
        title: const Text("Login"),
      ),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 420),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Icon(Icons.lock, size: 80),
                const SizedBox(height: 20),
                const Text(
                  "Aircraft Detector",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  status,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 30),

                TextField(
                  controller: usernameController,
                  decoration: const InputDecoration(
                    labelText: "Username",
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 16),

                TextField(
                  controller: passwordController,
                  obscureText: true,
                  decoration: const InputDecoration(
                    labelText: "Password",
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 20),

                ElevatedButton(
                  onPressed: loginWithPassword,
                  child: const Text("Login"),
                ),

                const SizedBox(height: 12),

                TextButton(
                  onPressed: () {
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(
                        builder: (_) => const HomeScreen(),
                      ),
                    );
                  },
                  child: const Text("Continue as Guest"),
                ),

                if (showBiometricButton) ...[
                  const SizedBox(height: 16),
                  OutlinedButton.icon(
                    onPressed: authenticateWithBiometrics,
                    icon: const Icon(Icons.fingerprint),
                    label: const Text("Use biometric / PIN instead"),
                  ),
                ],

                const SizedBox(height: 20),
                const Text(
                  "Demo login:\nusername: admin\npassword: aircraft123",
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}