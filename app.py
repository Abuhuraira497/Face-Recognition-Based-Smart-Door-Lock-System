"""Unified CLI for face-recognition door system with face_recognition library
with Electric Lock + Servo Motor Sequential Control

Optimized for weak processors: Reduced FPS, streamlined processing
Commands:
    capture  - capture face images for a person
    train    - train recognizer from existing dataset/
    recognize- run live recognition with annotations
    run      - live monitor: recognize, unlock known, deny unknown

Usage examples:
    python app.py capture --name Alice --count 40
    python app.py train
    python app.py recognize
    python app.py run
"""
import argparse
import os
import time
import pickle
import cv2
import uuid
import numpy as np
from datetime import datetime
import warnings
import hashlib
import threading
import queue
import serial
import serial.tools.list_ports
from collections import defaultdict
warnings.filterwarnings('ignore')

# Import face_recognition (should be available now)
try:
    import face_recognition
    FACE_RECOG_AVAILABLE = True
    print("✓ face_recognition library loaded successfully")
except ImportError:
    FACE_RECOG_AVAILABLE = False
    print("✗ face_recognition library not available")
    exit(1)

# Import for voice functionality
PYTTSX3_AVAILABLE = False
WIN32COM_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
    print("✓ pyttsx3 library loaded successfully")
    TTS_AVAILABLE = True
except ImportError:
    try:
        import win32com.client
        WIN32COM_AVAILABLE = True
        print("✓ win32com.client loaded successfully")
        TTS_AVAILABLE = True
    except ImportError:
        TTS_AVAILABLE = False
        print("✗ No TTS library available (install pyttsx3 or pywin32)")

# Import for Arduino serial communication
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
    print("✓ pyserial library loaded successfully")
except ImportError:
    SERIAL_AVAILABLE = False
    print("✗ pyserial not available (install pyserial for Arduino control)")

BASE_DIR = os.path.dirname(__file__)

# ============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# ============================================================================

# Reduced FPS for weak processors
TARGET_FPS = 5  # Reduced from ~30 to 5 FPS
FRAME_SKIP = 3   # Process 1 out of every 4 frames
RESIZE_FACTOR = 0.5  # Resize frames to half size for processing

# Face recognition settings
FACE_DETECTION_INTERVAL = 2  # Seconds between face detection runs
ENCODING_INTERVAL = 1  # Seconds between face encoding attempts

# Door control settings
DOOR_COOLDOWN = 15  # Increased cooldown to prevent rapid door operations

# ============================================================================
# ARDUINO CONTROLLER CLASS - OPTIMIZED
# ============================================================================

class ArduinoController:
    """Working Arduino controller with proper serial communication."""
    
    def __init__(self, port=None):
        self.port = port or 'COM4'
        self.serial_conn = None
        self.connected = False
        self.command_queue = queue.Queue()
        self.is_running = True
        
        # Start command processing thread
        self.command_thread = threading.Thread(target=self._process_commands, daemon=True)
        self.command_thread.start()
    
    def connect(self):
        """Connect to Arduino with proper error handling."""
        try:
            print(f"\n{'='*60}")
            print(f"CONNECTING TO ARDUINO ON {self.port}")
            print(f"{'='*60}")
            
            # Close existing connection
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                time.sleep(0.5)
            
            # Open new connection
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=9600,
                timeout=2,
                write_timeout=2,
                exclusive=True
            )
            
            print(f"✓ Port {self.port} opened")
            
            # Wait for Arduino to initialize
            print("Waiting for Arduino to reset...")
            time.sleep(3)
            
            # Clear buffers
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            # Test communication
            print("Testing communication...")
            
            # Send PING command
            self._send_raw("PING")
            
            # Read response
            response = self._read_response(timeout=3)
            print(f"Arduino response: '{response}'")
            
            # ACCEPT BOTH "PONG" AND "READY" AS VALID RESPONSES
            if response and ("PONG" in response.upper() or "READY" in response.upper()):
                print("✓ Arduino connected and responding!")
                self.connected = True
                
                # Test door sequence command
                print("\nTesting door sequence command...")
                self._send_raw("STATUS")
                status_response = self._read_response(timeout=2)
                if status_response:
                    print(f"Status test response: {status_response}")
                    
                return True
            else:
                print("✗ Arduino not responding correctly")
                self.connected = False
                return False
                
        except serial.SerialException as e:
            print(f"✗ Serial connection error: {e}")
            self.connected = False
            return False
        except Exception as e:
            print(f"✗ Connection error: {e}")
            self.connected = False
            return False
     
    def _send_raw(self, command):
        """Send raw command to Arduino."""
        if not self.serial_conn or not self.serial_conn.is_open:
            print(f"[SIM] Would send: {command}")
            return False
        
        try:
            # Add newline terminator
            full_command = command.strip() + '\n'
            self.serial_conn.write(full_command.encode('utf-8'))
            self.serial_conn.flush()
            return True
        except Exception as e:
            print(f"Send error: {e}")
            self.connected = False
            return False
    
    def _read_response(self, timeout=2):
        """Read response from Arduino."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return None
        
        try:
            # Set timeout
            original_timeout = self.serial_conn.timeout
            self.serial_conn.timeout = timeout
            
            # Read line
            response = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
            
            # Restore timeout
            self.serial_conn.timeout = original_timeout
            
            return response if response else None
            
        except Exception as e:
            print(f"Read error: {e}")
            return None
    
    def _process_commands(self):
        """Process commands from queue."""
        while self.is_running:
            try:
                command = self.command_queue.get(timeout=0.5)
                
                if self.connected:
                    # Send command
                    self._send_raw(command)
                    
                    # Read response if available
                    time.sleep(0.1)
                    if self.serial_conn.in_waiting:
                        response = self._read_response(timeout=1)
                        if response:
                            print(f"Arduino: {response}")
                else:
                    print(f"[SIM] Command: {command}")
                
                self.command_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Command processing error: {e}")
    
    def send_command(self, command):
        """Send command via queue."""
        self.command_queue.put(command)
        return True
    
    def open_door_sequence(self):
        """Start door sequence ONLY FOR KNOWN FACES."""
        print("[DOOR] Starting door sequence...")
        success = self.send_command("OPENDOOR")
        
        if success and self.connected:
            print("[DOOR] Door sequence command sent")
            
            # Wait and read responses for door sequence
            print("[DOOR] Waiting for door sequence to complete...")
            for i in range(15):  # Wait up to 15 seconds
                if self.serial_conn.in_waiting:
                    response = self._read_response(timeout=1)
                    if response:
                        print(f"[DOOR] {response}")
                        if "COMPLETE" in response.upper():
                            print("[DOOR] Door sequence completed successfully")
                            return True
                time.sleep(1)
            
            print("[DOOR] Door sequence may have completed")
            return True
        else:
            print("[DOOR SIMULATION] Door sequence would run")
            print("[DOOR SIMULATION] 1. Unlock → 2. Open → 3. Wait → 4. Close → 5. Lock")
            return False
    
    def test_connection(self):
        """Test Arduino connection."""
        if not self.connected:
            print("Arduino not connected")
            return False
        
        print("Testing Arduino connection...")
        self.send_command("PING")
        
        # Wait for response
        time.sleep(1)
        
        if self.serial_conn.in_waiting:
            response = self._read_response(timeout=2)
            if response and ("PONG" in response.upper() or "READY" in response.upper()):
                print("✓ Connection test passed")
                return True
        
        print("✗ Connection test failed")
        return False
    
    def manual_test_lock(self):
        """Test lock manually."""
        print("Testing lock...")
        self.send_command("LOCK_TEST")
    
    def manual_test_servo(self):
        """Test servo manually."""
        print("Testing servo...")
        self.send_command("SERVO_TEST")
    
    def run_complete_test(self):
        """Run complete test."""
        print("Running complete test...")
        self.send_command("TEST")
    
    def get_status(self):
        """Get status."""
        self.send_command("STATUS")
    
    def stop_door(self):
        """Emergency stop."""
        print("Emergency stop")
        self.send_command("STOP")
    
    def disconnect(self):
        """Disconnect from Arduino."""
        self.is_running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Disconnected from Arduino")

# Initialize Arduino controller (don't auto-connect)
arduino_controller = None
if SERIAL_AVAILABLE:
    try:
        print("\n" + "="*60)
        print("ARDUINO CONTROLLER AVAILABLE")
        print("="*60)
        print("Arduino will connect only when needed")
        print("="*60)
        # Create controller but don't connect
        arduino_controller = ArduinoController()
    except Exception as e:
        print(f"Failed to initialize Arduino controller: {e}")
        arduino_controller = None

# ============================================================================
# VOICE ANNOUNCER CLASS - OPTIMIZED
# ============================================================================

class VoiceAnnouncer:
    """Handles voice announcements for access granted/denied."""
    
    def __init__(self):
        self.tts_engine = None
        self.tts_type = None
        self.announcement_queue = queue.Queue()
        self.is_running = True
        self.last_announcement = {}
        self.cooldown_seconds = 10  # Increased cooldown to reduce CPU usage
        
        # Try pyttsx3 first
        if PYTTSX3_AVAILABLE:
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_type = 'pyttsx3'
                
                # Configure engine
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                
                # Try to get a working voice
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    # Try to use first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)
                
                print(f"✓ Voice announcer initialized (pyttsx3)")
                print(f"  Rate: 150, Volume: 0.9")
                
            except Exception as e:
                print(f"✗ pyttsx3 failed: {e}")
                self.tts_engine = None
        
        # If pyttsx3 failed, try Windows SAPI
        if self.tts_engine is None and WIN32COM_AVAILABLE:
            try:
                import win32com.client
                self.tts_engine = win32com.client.Dispatch("SAPI.SpVoice")
                self.tts_type = 'win32com'
                print("✓ Voice announcer initialized (Windows SAPI)")
            except Exception as e:
                print(f"✗ Windows SAPI failed: {e}")
                self.tts_engine = None
        
        # Check if we have a working TTS engine
        if self.tts_engine:
            print("✓ Voice announcements ENABLED")
            # Start announcement thread
            self.announcement_thread = threading.Thread(target=self._process_announcements, daemon=True)
            self.announcement_thread.start()
        else:
            print("⚠️  Voice announcements DISABLED - no TTS engine available")
    
    def _process_announcements(self):
        """Process announcements from queue."""
        while self.is_running:
            try:
                # Get announcement from queue with timeout
                try:
                    announcement_type, name = self.announcement_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Check cooldown
                current_time = time.time()
                announcement_key = f"{announcement_type}_{name}"
                
                if announcement_key in self.last_announcement:
                    time_since_last = current_time - self.last_announcement[announcement_key]
                    if time_since_last < self.cooldown_seconds:
                        self.announcement_queue.task_done()
                        continue
                
                # Speak announcement
                self._speak(announcement_type, name)
                
                # Update last announcement time
                self.last_announcement[announcement_key] = current_time
                
                self.announcement_queue.task_done()
                
            except Exception as e:
                print(f"Announcement error: {e}")
    
    def _speak(self, announcement_type, name=None):
        """Speak the announcement."""
        try:
            if announcement_type == "access_granted":
                text = f"Access granted for {name}" if name and name != "Unknown" else "Access granted"
            elif announcement_type == "access_denied":
                text = "Access denied"
            elif announcement_type == "intruder":
                text = "Intruder detected"
            elif announcement_type == "door_sequence_start":
                text = "Door opening"
            else:
                text = "System alert"
            
            print(f"[VOICE] {text}")
            
            if self.tts_type == 'pyttsx3':
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            elif self.tts_type == 'win32com':
                self.tts_engine.Speak(text)
                
        except Exception as e:
            print(f"Speech error: {e}")
    
    def announce_access_granted(self, name="Unknown"):
        """Announce access granted."""
        if self.tts_engine:
            self.announcement_queue.put(("access_granted", name))
    
    def announce_access_denied(self):
        """Announce access denied."""
        if self.tts_engine:
            self.announcement_queue.put(("access_denied", "unknown"))
    
    def announce_intruder(self):
        """Announce intruder detected."""
        if self.tts_engine:
            self.announcement_queue.put(("intruder", "intruder"))
    
    def announce_door_sequence_start(self):
        """Announce door sequence start."""
        if self.tts_engine:
            self.announcement_queue.put(("door_sequence_start", None))
    
    def stop(self):
        """Stop the announcer."""
        self.is_running = False

# Initialize voice announcer
voice_announcer = VoiceAnnouncer()

# ============================================================================
# ARDUINO TEST FUNCTIONS
# ============================================================================

def test_arduino_connection():
    """Test Arduino connection."""
    if not SERIAL_AVAILABLE:
        print("ERROR: pyserial not installed. Run: pip install pyserial")
        return False
    
    global arduino_controller
    if arduino_controller is None:
        print("Initializing Arduino controller...")
        arduino_controller = ArduinoController()
    
    # Try to connect
    if arduino_controller.connect():
        print("✓ Arduino is connected")
        
        # Test communication
        success = arduino_controller.test_connection()
        
        if success:
            print("✓ Arduino communication test successful")
            voice_announcer.announce_access_granted("Test")  # Test voice
        else:
            print("✗ Arduino communication test failed")
        
        return success
    else:
        print("\n✗ Arduino not connected")
        print("\n" + "="*60)
        print("TROUBLESHOOTING STEPS:")
        print("="*60)
        print("1. ✅ Check Arduino is connected via USB")
        print("2. ✅ Upload the Arduino door control code")
        print("3. ⚠️  CLOSE Arduino IDE (most common issue!)")
        print("4. ⚠️  CLOSE Serial Monitor")
        print("5. 🔄 Disconnect/reconnect Arduino USB")
        print("6. 💻 Run VS Code as Administrator")
        print("7. 🔧 Try specifying port manually: --port COM3")
        print("8. 📋 Check Device Manager for correct COM port")
        print("="*60)
        
        # List available ports
        if SERIAL_AVAILABLE:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            print("\nAvailable COM ports:")
            for port in ports:
                print(f"  {port.device}: {port.description}")
        
        # Try to connect manually
        retry = input("\nTry to connect manually? (y/n): ").strip().lower()
        if retry == 'y':
            port = input("Enter COM port (e.g., COM4): ").strip()
            arduino_controller = ArduinoController(port=port)
            return arduino_controller.connect()
        
        return False

def arduino_manual_test():
    """Manual test of Arduino functions."""
    if not test_arduino_connection():
        return
    
    print("\nArduino Door Control Test (Electric Lock + Servo)")
    print("=" * 50)
    
    while True:
        print("\nTest Options:")
        print("1. Start full door sequence")
        print("2. Emergency stop")
        print("3. Test electric lock only")
        print("4. Test servo only")
        print("5. Get system status")
        print("6. Run complete test sequence")
        print("0. Return to main menu")
        
        choice = input('\nSelect test (0-6): ').strip()
        
        if choice == '0':
            break
        elif choice == '1':
            arduino_controller.open_door_sequence()
        elif choice == '2':
            arduino_controller.stop_door()
            print("Emergency stop command sent")
        elif choice == '3':
            print("Testing electric lock...")
            arduino_controller.manual_test_lock()
        elif choice == '4':
            print("Testing servo motor...")
            arduino_controller.manual_test_servo()
        elif choice == '5':
            arduino_controller.get_status()
        elif choice == '6':
            print("Running complete test sequence...")
            arduino_controller.run_complete_test()
        else:
            print("Invalid choice")

def test_arduino_door():
    """Test Arduino door sequence."""
    print("\n" + "="*60)
    print("ARDUINO DOOR TEST")
    print("="*60)
    
    if not SERIAL_AVAILABLE:
        print("ERROR: pyserial not installed")
        return
    
    global arduino_controller
    
    # Initialize if needed
    if arduino_controller is None:
        arduino_controller = ArduinoController()
    
    # Try to connect
    if arduino_controller.connect():
        print("✓ Arduino connected")
        
        # Test door sequence
        print("\nTesting door sequence...")
        success = arduino_controller.open_door_sequence()
        
        if success:
            print("✓ Door sequence completed successfully")
            voice_announcer.announce_door_sequence_start()
        else:
            print("✗ Door sequence failed")
    else:
        print("✗ Could not connect to Arduino")
    
    print("="*60)

# ============================================================================
# DOOR CONTROL with Arduino Integration - FIXED & OPTIMIZED
# ============================================================================

def get_door_control():
    """Door control interface with Arduino."""
    class DoorControl:
        def __init__(self):
            self.last_unlock_time = 0
            self.unlock_cooldown = DOOR_COOLDOWN  # Use global cooldown
            self.last_unlock_for_person = {}  # Track last unlock per person
            self.arduino = arduino_controller
            
        def unlock(self, name="Unknown", duration=5):
            current_time = time.time()
            
            # CRITICAL FIX: NEVER unlock for unknown persons
            if name == "Unknown" or name.startswith("Person_"):
                print(f"[DOOR] Access DENIED for unknown: {name}")
                return False
            
            # Check person-specific cooldown
            if name in self.last_unlock_for_person:
                time_since_last = current_time - self.last_unlock_for_person[name]
                if time_since_last < self.unlock_cooldown:
                    print(f"[DOOR] Cooldown active for {name}. Next unlock in {self.unlock_cooldown - time_since_last:.1f}s")
                    return False
            
            # Check general cooldown
            if current_time - self.last_unlock_time < 2:  # 2 seconds minimum between any unlocks
                print(f"[DOOR] System cooldown active. Please wait...")
                return False
            
            print(f"\n{'='*60}")
            print(f"ACCESS GRANTED: {name}")
            print(f"{'='*60}")
            
            # Send command to Arduino to start door sequence
            door_sequence_started = False
            if self.arduino and self.arduino.connected:
                print(f"[DOOR] Starting door sequence for {name}")
                door_sequence_started = self.arduino.open_door_sequence()
            else:
                print(f"[DOOR SIMULATION] Door sequence for {name}")
                print(f"[DOOR SIMULATION] 1. Unlock → 2. Open → 3. Wait → 4. Close → 5. Lock")
                door_sequence_started = True
            
            if door_sequence_started:
                # Update timers
                self.last_unlock_time = current_time
                self.last_unlock_for_person[name] = current_time
                
                # Announce door sequence start
                voice_announcer.announce_door_sequence_start()
                
                return True
            else:
                print(f"[DOOR] Failed to start door sequence")
                return False
        
        def lock(self):
            print("[DOOR] Emergency lock command")
            if self.arduino and self.arduino.connected:
                self.arduino.stop_door()
    
    return DoorControl()

# ============================================================================
# INTRUDER TRACKER CLASS - OPTIMIZED
# ============================================================================

class IntruderTracker:
    """Track intruders to avoid duplicate image saving."""
    
    def __init__(self, cooldown_seconds=30):
        self.cooldown_seconds = cooldown_seconds
        self.tracked_intruders = {}  # face_encoding -> last_saved_time
    
    def get_face_hash(self, face_encoding):
        """Create a hash from face encoding for tracking."""
        if face_encoding is None or len(face_encoding) == 0:
            return None
        
        # Use first 5 values for faster hash
        encoding_str = ','.join([str(x) for x in face_encoding[:5]])
        return hashlib.md5(encoding_str.encode()).hexdigest()[:12]
    
    def should_save_intruder(self, face_encoding):
        """Check if we should save this intruder (not saved recently)."""
        if face_encoding is None:
            return False
        
        face_hash = self.get_face_hash(face_encoding)
        if face_hash is None:
            return False
        
        current_time = time.time()
        
        if face_hash in self.tracked_intruders:
            last_saved = self.tracked_intruders[face_hash]
            if current_time - last_saved < self.cooldown_seconds:
                return False
        
        # Update tracking
        self.tracked_intruders[face_hash] = current_time
        return True

# Initialize intruder tracker
intruder_tracker = IntruderTracker(cooldown_seconds=30)

# ============================================================================
# FACE RECOGNIZER CLASS using face_recognition - OPTIMIZED
# ============================================================================

class FaceRecognizer:
    """Face recognizer using face_recognition library."""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(BASE_DIR, 'model.pkl')
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_labels = []
        self.label_to_name = {}
        self.name_to_label = {}
        self.next_label = 0
        
        # Load existing model if it exists
        self.load(self.model_path)
    
    def train_from_dataset(self, dataset_path):
        """Train from existing dataset folder structure using face_recognition."""
        print(f"\n{'='*60}")
        print("TRAINING WITH FACE_RECOGNITION LIBRARY")
        print(f"{'='*60}")
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset directory not found: {dataset_path}")
            print("Please run 'capture' command first to create a dataset.")
            return False
        
        # Load all images from dataset
        person_dirs = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
        
        if not person_dirs:
            print(f"ERROR: No person directories found in {dataset_path}")
            print("Please run 'capture' command first to add people.")
            return False
        
        print(f"Found {len(person_dirs)} persons in dataset:")
        
        total_images_processed = 0
        total_faces_encoded = 0
        
        # Clear existing data
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_labels = []
        self.label_to_name = {}
        self.name_to_label = {}
        self.next_label = 0
        
        for person_name in sorted(person_dirs):
            person_dir = os.path.join(dataset_path, person_name)
            
            # Assign label ID
            if person_name not in self.name_to_label:
                self.name_to_label[person_name] = self.next_label
                self.label_to_name[self.next_label] = person_name
                self.next_label += 1
            
            label_id = self.name_to_label[person_name]
            
            # Load all images for this person
            image_files = [f for f in os.listdir(person_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"\n  Processing {person_name}: {len(image_files)} images")
            
            if not image_files:
                print(f"    Warning: No images found for {person_name}")
                continue
            
            # Limit to 20 images per person for training
            image_files = image_files[:20]
            encoded_for_person = 0
            
            for i, fname in enumerate(image_files):
                img_path = os.path.join(person_dir, fname)
                
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Resize for faster processing
                    small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                    
                    # Convert BGR to RGB
                    rgb_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                    
                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(rgb_img)
                    
                    if len(face_encodings) > 0:
                        # Use the first face encoding
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(person_name)
                        self.known_face_labels.append(label_id)
                        encoded_for_person += 1
                        total_faces_encoded += 1
                    
                    total_images_processed += 1
                    
                except Exception as e:
                    continue
            
            print(f"    Successfully encoded {encoded_for_person} faces for {person_name}")
        
        if len(self.known_face_encodings) == 0:
            print("\nERROR: No faces could be encoded!")
            return False
        
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {total_images_processed}")
        print(f"Total faces encoded: {len(self.known_face_encodings)}")
        print(f"Number of persons: {len(person_dirs)}")
        
        # Save model
        self._save_model()
        
        return True
    
    def _save_model(self):
        """Save trained model to file."""
        try:
            data = {
                'known_face_encodings': self.known_face_encodings,
                'known_face_names': self.known_face_names,
                'known_face_labels': self.known_face_labels,
                'label_to_name': self.label_to_name,
                'name_to_label': self.name_to_label,
                'next_label': self.next_label
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"\n✓ Model saved to: {self.model_path}")
            print(f"  Face encodings: {len(self.known_face_encodings)}")
            print(f"  Persons: {len(self.name_to_label)}")
            
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def predict_with_encoding(self, face_image):
        """Predict label and confidence and also return face encoding."""
        try:
            if len(self.known_face_encodings) == 0:
                return -1, float('inf'), None
            
            # Resize face for faster processing
            face_image = cv2.resize(face_image, (100, 100))
            
            # Convert to RGB
            if len(face_image.shape) == 2:  # Grayscale
                rgb_face = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            else:  # BGR
                rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Get encoding for the face
            face_encodings = face_recognition.face_encodings(rgb_face)
            
            if not face_encodings:
                return -1, float('inf'), None
            
            # Compare with known faces
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, 
                face_encodings[0]
            )
            
            # Find best match
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]
            
            # Return label, distance, and face encoding
            return self.known_face_labels[best_match_index], min_distance, face_encodings[0]
            
        except Exception as e:
            return -1, float('inf'), None
    
    def load(self, path):
        """Load model from file."""
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    
                    self.known_face_encodings = data['known_face_encodings']
                    self.known_face_names = data['known_face_names']
                    self.known_face_labels = data['known_face_labels']
                    self.label_to_name = data['label_to_name']
                    self.name_to_label = data['name_to_label']
                    self.next_label = data['next_label']
                    
                    print(f"✓ Loaded face_recognition model from {path}")
                    print(f"  Face encodings: {len(self.known_face_encodings)}")
                    print(f"  Persons: {len(self.name_to_label)}")
                    
                    return True
            else:
                print(f"⚠️  No saved model found at {path}")
                return False
                    
        except Exception as e:
            print(f"Load error: {e}")
            return False

# Initialize recognizer
recognizer = FaceRecognizer()

# ============================================================================
# HELPER FUNCTIONS - OPTIMIZED
# ============================================================================

def interpret_score(confidence, threshold=0.6):
    """Return (is_known:bool, percent_conf:float, text:str) for face_recognition."""
    if confidence == float('inf') or confidence < 0:
        return False, 0.0, 'n/a'
    
    dist = float(confidence)
    # STRICTER THRESHOLD: 0.5 instead of 0.6
    is_known = dist < 0.5  # More strict threshold
    
    # Convert distance to percentage
    if dist <= 0.0:
        percent = 100.0
    elif dist >= threshold:
        percent = 0.0
    else:
        percent = 100.0 * (1.0 - (dist / threshold))
    
    text = f'{dist:.3f}'
    
    return is_known, percent, text


def draw_label(frame, x, y, w, h, text, color=(0,255,0), thickness=2):
    """Draw face rectangle with label - OPTIMIZED."""
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    
    label_y = max(y - 5, th + 5)
    cv2.rectangle(frame, (x, label_y - th - 5), (x + tw + 10, label_y), color, -1)
    cv2.putText(frame, text, (x + 5, label_y - 5), font, font_scale, (255, 255, 255), 1)

# ============================================================================
# OPTIMIZED CAMERA FUNCTIONS
# ============================================================================

def get_camera(max_retries=3):
    """Get camera with error handling and performance optimization."""
    for retry in range(max_retries):
        for i in range(3):  # Try only first 3 indices
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    # Set lower resolution for weak processors
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                    
                    # Test frame read
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✓ Camera found at index {i} (640x480 @ {TARGET_FPS}FPS)")
                        return cap, False  # False = not simulation mode
                    else:
                        cap.release()
            except:
                pass
    
    print("⚠️  Camera not accessible - Using simulation mode")
    return None, True

# ============================================================================
# OPTIMIZED MAIN FUNCTIONS
# ============================================================================

def capture_images(name: str, count: int = 20, delay: float = 0.5):  # Reduced count and increased delay
    """Capture new face images with face_recognition detection - OPTIMIZED."""
    out_dir = os.path.join(BASE_DIR, 'dataset', name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Get camera
    cap, simulation_mode = get_camera()
    if simulation_mode:
        print("Cannot capture without camera")
        return
    
    saved = 0
    print(f"\n{'='*60}")
    print(f"CAPTURING {count} IMAGES FOR '{name.upper()}'")
    print(f"{'='*60}")
    print("Instructions:")
    print("1. Face the camera directly")
    print("2. Ensure good lighting")
    print("3. Show natural expression")
    print("4. Press 'q' to quit, 's' to save manually")
    print(f"{'='*60}")
    
    frame_count = 0
    
    while saved < count:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for performance
        if frame_count % 3 != 0:
            continue
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # Convert to RGB for face_recognition
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        # Display
        display = frame.copy()
        for face_loc in face_locations:
            try:
                if len(face_loc) >= 4:
                    top, right, bottom, left = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                    # Scale back up face locations since the frame was scaled to 0.5
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2
                    cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
            except:
                continue
        
        cv2.putText(display, f"Capturing: {name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Saved: {saved}/{count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press 'q' to quit, 's' to save manually", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
        
        cv2.imshow('Capture Face Images', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s') or (len(face_locations) > 0 and saved < count):
            if len(face_locations) > 0:
                try:
                    # Get largest face
                    face_loc = max(face_locations, 
                        key=lambda loc: (loc[1]-loc[3]) * (loc[2]-loc[0]) if len(loc) >= 4 else 0)
                    
                    if len(face_loc) >= 4:
                        top, right, bottom, left = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                        # Scale back up
                        top *= 2
                        right *= 2
                        bottom *= 2
                        left *= 2
                        
                        face_img = frame[top:bottom, left:right]
                        
                        if face_img.size > 0:
                            # Resize for consistency
                            face_img = cv2.resize(face_img, (200, 200))
                            ts = int(time.time() * 1000)
                            fname = os.path.join(out_dir, f'{name}_{ts}.jpg')
                            cv2.imwrite(fname, face_img)
                            saved += 1
                            print(f"✓ Saved {saved}/{count}")
                            
                            time.sleep(delay)
                except:
                    print("Error saving face")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"CAPTURE COMPLETE")
    print(f"{'='*60}")
    print(f"Saved {saved} images for '{name}'")


def train_model(dataset_path=None, model_path=None):
    """Train model from EXISTING dataset using face_recognition."""
    dataset_path = dataset_path or os.path.join(BASE_DIR, 'dataset')
    model_path = model_path or os.path.join(BASE_DIR, 'model.pkl')
    
    print(f"\n{'='*60}")
    print("TRAINING WITH FACE_RECOGNITION")
    print(f"{'='*60}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at: {dataset_path}")
        return False
    
    # Check for person directories
    person_dirs = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not person_dirs:
        print(f"ERROR: No person directories found in {dataset_path}")
        return False
    
    print(f"Dataset location: {dataset_path}")
    print(f"Found {len(person_dirs)} persons")
    
    # Train the recognizer
    success = recognizer.train_from_dataset(dataset_path)
    
    if not success:
        print("\nTraining failed!")
        return False
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    
    return True


def recognize_loop(model_path=None, threshold=None):
    """Test recognition with webcam using face_recognition - OPTIMIZED."""
    model_path = model_path or os.path.join(BASE_DIR, 'model.pkl')
    
    # Check if model is loaded
    if len(recognizer.known_face_encodings) == 0:
        print("ERROR: No trained model found!")
        return
    
    print(f"\n{'='*60}")
    print("LIVE FACE RECOGNITION TEST")
    print(f"{'='*60}")
    print("Press 'q' to quit")
    
    # Get camera
    cap, simulation_mode = get_camera()
    if simulation_mode:
        print("Cannot recognize without camera")
        return
    
    # Default threshold - STRICTER
    if threshold is None:
        threshold = 0.5  # Stricter threshold
    
    frame_count = 0
    last_detection_time = 0
    face_locations_cache = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for performance
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        
        current_time = time.time()
        
        # Detect faces less frequently
        if current_time - last_detection_time > FACE_DETECTION_INTERVAL:
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations_cache = face_recognition.face_locations(rgb_small_frame)
            last_detection_time = current_time
        
        known_faces_count = 0
        unknown_faces_count = 0
        
        for face_loc in face_locations_cache:
            try:
                if len(face_loc) >= 4:
                    top, right, bottom, left = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                    # Scale back up
                    top = int(top / RESIZE_FACTOR)
                    right = int(right / RESIZE_FACTOR)
                    bottom = int(bottom / RESIZE_FACTOR)
                    left = int(left / RESIZE_FACTOR)
                    
                    x, y, w, h = left, top, right - left, bottom - top
                    
                    # Extract face
                    face_img = frame[y:y+h, x:x+w]
                    
                    if face_img.size == 0:
                        continue
                        
                    face_img_resized = cv2.resize(face_img, (100, 100))
                    
                    # Predict with encoding
                    label, confidence, face_encoding = recognizer.predict_with_encoding(face_img_resized)
                    is_known, percent, score_text = interpret_score(confidence, threshold)
                    
                    if is_known and label != -1:
                        name = recognizer.label_to_name.get(label, f'Person_{label}')
                        color = (0, 255, 0)
                        text = f"{name}"
                        known_faces_count += 1
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)
                        text = f"Unknown"
                        unknown_faces_count += 1
                    
                    draw_label(frame, x, y, w, h, text, color)
            except:
                continue
        
        # Display info
        cv2.putText(frame, f"FPS: {TARGET_FPS}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces: Known: {known_faces_count}, Unknown: {unknown_faces_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Face Recognition Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nRecognition test stopped.")


def run_live(model_path=None, threshold=None, port=None):
    """Run door monitor using face_recognition with Arduino door control - OPTIMIZED."""
    model_path = model_path or os.path.join(BASE_DIR, 'model.pkl')
    
    # Check if model is loaded
    if len(recognizer.known_face_encodings) == 0:
        print("ERROR: No trained model found!")
        return
    
    # Check if Arduino needs to be initialized
    if port and SERIAL_AVAILABLE:
        global arduino_controller
        arduino_controller = ArduinoController(port=port)
    
    print(f"\n{'='*60}")
    print("DOOR MONITOR ACTIVE - OPTIMIZED MODE")
    print(f"{'='*60}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Frame skip: 1 in every {FRAME_SKIP} frames")
    print(f"Resolution: {int(640*RESIZE_FACTOR)}x{int(480*RESIZE_FACTOR)}")
    print("Press 'q' to quit")
    
    # ==================== ARDUINO CONNECTION ====================
    print(f"\n{'='*60}")
    print("CHECKING ARDUINO CONNECTION")
    print(f"{'='*60}")
    
    arduino_ok = False
    if arduino_controller:
        if not arduino_controller.connected:
            print("⚠️  Arduino not connected - attempting to connect...")
            arduino_ok = arduino_controller.connect()
        else:
            arduino_ok = True
            print("✓ Arduino already connected")
    else:
        print("⚠️  Arduino controller not initialized")
    
    if arduino_ok:
        print("✓ Arduino connected and ready for door control")
    else:
        print("⚠️  Arduino not connected")
        print("   Door will be simulated (no physical movement)")
    print(f"{'='*60}")
    
    # Get camera
    cap, simulation_mode = get_camera()
    
    # Default threshold - STRICTER
    if threshold is None:
        threshold = 0.5  # Stricter threshold
    
    door = get_door_control()
    
    # Intruder directory
    intruders_dir = os.path.join(BASE_DIR, 'intruders')
    os.makedirs(intruders_dir, exist_ok=True)
    
    last_intruder_saved = 0
    global_cooldown = 15  # Increased cooldown
    
    frame_count = 0
    last_detection_time = 0
    last_encoding_time = 0
    face_locations_cache = []
    face_encodings_cache = {}
    
    print("\nSystem ready. Waiting for faces...")
    
    while True:
        if not simulation_mode:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Simulation mode
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "SIMULATION MODE", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            ret = True
        
        frame_count += 1
        current_time = time.time()
        
        # Skip frames for performance
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        
        # Detect faces less frequently
        if current_time - last_detection_time > FACE_DETECTION_INTERVAL:
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations_cache = face_recognition.face_locations(rgb_small_frame)
            last_detection_time = current_time
            face_encodings_cache = {}  # Clear cache
        
        known_faces_detected = []
        unknown_faces_detected = 0
        access_granted_in_frame = False
        
        for idx, face_loc in enumerate(face_locations_cache):
            try:
                # Safely unpack face location
                if len(face_loc) >= 4:
                    top, right, bottom, left = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                    # Scale back up
                    top = int(top / RESIZE_FACTOR)
                    right = int(right / RESIZE_FACTOR)
                    bottom = int(bottom / RESIZE_FACTOR)
                    left = int(left / RESIZE_FACTOR)
                    
                    x, y, w, h = left, top, right - left, bottom - top
                    
                    # Extract face
                    face_img = frame[y:y+h, x:x+w]
                    
                    if face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                        continue
                    
                    # Get or compute face encoding less frequently
                    face_encoding = None
                    face_key = f"{x}_{y}"
                    
                    if face_key in face_encodings_cache:
                        face_encoding = face_encodings_cache[face_key]
                        # Get label and confidence
                        face_img_resized = cv2.resize(face_img, (100, 100))
                        label, confidence, _ = recognizer.predict_with_encoding(face_img_resized)
                    elif current_time - last_encoding_time > ENCODING_INTERVAL:
                        face_img_resized = cv2.resize(face_img, (100, 100))
                        label, confidence, face_encoding = recognizer.predict_with_encoding(face_img_resized)
                        if face_encoding is not None:
                            face_encodings_cache[face_key] = face_encoding
                        last_encoding_time = current_time
                    else:
                        continue
                    
                    if face_encoding is None:
                        continue
                    
                    is_known, percent, score_text = interpret_score(confidence, threshold)
                    
                    if is_known and label != -1:
                        name = recognizer.label_to_name.get(label, f'Person_{label}')
                        color = (0, 255, 0)
                        text = f"ACCESS: {name}"
                        
                        if name not in known_faces_detected:
                            known_faces_detected.append(name)
                        
                        # Try to unlock/open door ONLY for known faces
                        if not name.startswith("Person_"):
                            door_unlocked = door.unlock(name, duration=5)
                            
                            if door_unlocked:
                                access_granted_in_frame = True
                                voice_announcer.announce_access_granted(name)
                        else:
                            # This is actually unknown (but distance was close)
                            color = (0, 0, 255)
                            text = "ACCESS DENIED"
                            unknown_faces_detected += 1
                            
                    else:
                        color = (0, 0, 255)
                        text = "ACCESS DENIED"
                        unknown_faces_detected += 1
                        
                        # Check if we should save this intruder
                        should_save = False
                        
                        # Use face encoding tracker
                        if intruder_tracker.should_save_intruder(face_encoding):
                            should_save = True
                        
                        # Global cooldown (backup)
                        if not should_save and current_time - last_intruder_saved > global_cooldown:
                            should_save = True
                        
                        # Save intruder image if needed
                        if should_save and not simulation_mode:  # Only save in real mode
                            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                            intruder_path = os.path.join(intruders_dir, f'intruder_{ts}.jpg')
                            
                            # Save cropped face
                            face_cropped = frame[max(0, y-10):min(frame.shape[0], y+h+10), 
                                                max(0, x-10):min(frame.shape[1], x+w+10)]
                            
                            if face_cropped.size > 0:
                                cv2.imwrite(intruder_path, face_cropped)
                                print(f"\n[ALERT] Unknown person detected!")
                                last_intruder_saved = current_time
                                voice_announcer.announce_access_denied()
                    
                    draw_label(frame, x, y, w, h, text, color, thickness=3)
            except Exception as e:
                print(f"Face processing error: {e}")
                continue
        
        # Display status
        status_y = 40
        if access_granted_in_frame:
            status = "ACCESS GRANTED"
            status_color = (0, 255, 0)
        elif unknown_faces_detected > 0:
            status = "ACCESS DENIED"
            status_color = (0, 0, 255)
        elif len(known_faces_detected) > 0:
            status = f"ACCESS OK"
            status_color = (0, 255, 0)
        else:
            status = "NO FACES"
            status_color = (255, 255, 0)
        
        cv2.putText(frame, status, (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        
        # Show performance info
        cv2.putText(frame, f"FPS: {TARGET_FPS}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show Arduino status
        arduino_status = "ARDUINO: ONLINE" if arduino_ok else "ARDUINO: OFFLINE"
        arduino_color = (0, 255, 0) if arduino_ok else (0, 0, 255)
        cv2.putText(frame, arduino_status, (frame.shape[1] - 200, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, arduino_color, 2)
        
        if simulation_mode:
            cv2.putText(frame, "SIMULATION MODE", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Door Monitor - Optimized', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    if not simulation_mode and cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    
    print("\nDoor monitor stopped.")

# ============================================================================
# SIMPLIFIED INTERACTIVE MENU
# ============================================================================

def interactive_menu():
    """Interactive menu."""
    while True:
        print('\n' + '='*60)
        print('FACE RECOGNITION DOOR SYSTEM - OPTIMIZED')
        print('='*60)
        print('1. Capture new face images')
        print('2. Train model')
        print('3. Test recognition')
        print('4. Run door monitor')
        print('5. Test Arduino door')
        print('6. Arduino manual test')
        print('7. Check system status')
        print('0. Exit')
        print('='*60)
        
        choice = input('\nSelect option (0-7): ').strip()
        
        if choice == '1':
            name = input('Enter person name: ').strip()
            if not name:
                print("Name is required!")
                continue
            count = input('Number of images [20]: ').strip()
            count = int(count) if count else 20
            capture_images(name, count)
            
        elif choice == '2':
            train_model()
            
        elif choice == '3':
            recognize_loop()
            
        elif choice == '4':
            run_live()
            
        elif choice == '5':
            test_arduino_door()
            
        elif choice == '6':
            arduino_manual_test()
            
        elif choice == '7':
            print("\nSystem Status:")
            print(f"Target FPS: {TARGET_FPS}")
            print(f"Frame skip: 1/{FRAME_SKIP}")
            print(f"Resolution scale: {RESIZE_FACTOR}")
            print(f"Recognition threshold: 0.5 (strict)")
            print(f"Door cooldown: {DOOR_COOLDOWN} seconds")
            
            # Arduino status
            if arduino_controller:
                print(f"Arduino: {'Connected' if arduino_controller.connected else 'Disconnected'}")
                if arduino_controller.connected:
                    print(f"Arduino port: {arduino_controller.port}")
            else:
                print("Arduino: Not initialized")
            
            # Check model
            if len(recognizer.known_face_encodings) > 0:
                print(f"Model loaded: {len(recognizer.known_face_encodings)} faces")
                print(f"Persons: {list(recognizer.name_to_label.keys())}")
            else:
                print("Model: Not trained")
            
            # Voice status
            print(f"Voice: {'Enabled' if voice_announcer.tts_engine else 'Disabled'}")
            
            input("\nPress Enter to continue...")
                
        elif choice == '0':
            print("\nShutting down...")
            break
            
        else:
            print("\nInvalid option.")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Face Recognition Door System - Optimized')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Capture
    capture_parser = subparsers.add_parser('capture', help='Capture new face images')
    capture_parser.add_argument('--name', required=True, help='Person name')
    capture_parser.add_argument('--count', type=int, default=20, help='Number of images')
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train from existing dataset')
    
    # Recognize
    rec_parser = subparsers.add_parser('recognize', help='Test recognition')
    
    # Run
    run_parser = subparsers.add_parser('run', help='Run door monitor')
    run_parser.add_argument('--port', help='Arduino COM port (e.g., COM3)')
    
    # Arduino test
    arduino_parser = subparsers.add_parser('arduino-test', help='Test Arduino connection')
    arduino_parser.add_argument('--port', help='Arduino COM port')
    
    args = parser.parse_args()
    
    if args.command == 'capture':
        capture_images(args.name, args.count)
    elif args.command == 'train':
        train_model()
    elif args.command == 'recognize':
        recognize_loop()
    elif args.command == 'run':
        run_live(port=args.port)
    elif args.command == 'arduino-test':
        if args.port and SERIAL_AVAILABLE:
            global arduino_controller
            arduino_controller = ArduinoController(port=args.port)
        test_arduino_connection()
    else:
        # Interactive mode
        interactive_menu()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Cleanup
        if arduino_controller:
            arduino_controller.disconnect()
        voice_announcer.stop()
        cv2.destroyAllWindows()
        print("\nSystem shutdown complete.")