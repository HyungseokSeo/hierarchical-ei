"""
Real-time demonstration of Hierarchical Emotional Intelligence Model

This script provides:
- Webcam-based emotion recognition
- Real-time emotional state tracking
- Visualization of hierarchical processing
- Interactive emotional intelligence demonstration
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import librosa
import sounddevice as sd
from collections import deque
import time
import argparse
from pathlib import Path
import threading
import queue

from hierarchical_ei_model import HierarchicalEmotionalIntelligence, HierarchicalConfig


class RealTimeDemo:
    """Real-time demonstration of Hierarchical EI"""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        # Load model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path, config_path)
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Audio capture
        self.audio_queue = queue.Queue()
        self.audio_sample_rate = 16000
        
        # Buffers for temporal processing
        self.frame_buffer = deque(maxlen=300)  # 10 seconds at 30fps
        self.audio_buffer = deque(maxlen=10 * self.audio_sample_rate)  # 10 seconds
        
        # State tracking
        self.emotion_history = deque(maxlen=100)
        self.mood_history = deque(maxlen=50)
        self.current_emotion = "Neutral"
        self.confidence = 0.0
        
        # Visualization
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 
                              'Sad', 'Surprise', 'Neutral', 'Contempt']
        self.emotion_colors = ['red', 'green', 'purple', 'yellow', 
                              'blue', 'orange', 'gray', 'brown']
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        
    def load_model(self, model_path: str, config_path: str):
        """Load trained model"""
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = HierarchicalConfig(**config_dict.get('model', {}))
        self.model = HierarchicalEmotionalIntelligence(config)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
        
    def start_audio_stream(self):
        """Start audio capture"""
        self.audio_stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.audio_sample_rate
        )
        self.audio_stream.start()
        
    def preprocess_frame(self, frame):
        """Preprocess video frame for model"""
        # Resize and normalize
        frame_resized = cv2.resize(frame, (64, 64))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)
        return frame_tensor
        
    def preprocess_audio(self, audio_data):
        """Preprocess audio for model"""
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.audio_sample_rate,
            n_mels=128
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_norm = (mel_spec_db + 80) / 80
        
        return torch.from_numpy(mel_spec_norm).float()
        
    @torch.no_grad()
    def process_hierarchical(self):
        """Process through hierarchical model"""
        if len(self.frame_buffer) < 30:  # Need at least 1 second
            return None
            
        # Prepare batch
        frames = list(self.frame_buffer)[-30:]  # Last 30 frames
        visual_batch = torch.stack([self.preprocess_frame(f) for f in frames])
        visual_batch = visual_batch.unsqueeze(0).to(self.device)
        
        # Mock audio for demo (replace with actual audio processing)
        audio_batch = torch.randn(1, 30, 128).to(self.device)
        
        # Context (could include time of day, location, etc.)
        context = torch.randn(1, 128).to(self.device)
        
        # Forward pass
        outputs = self.model(visual_batch, audio_batch, context, return_all=False)
        
        return outputs
        
    def update_emotion_display(self, frame, outputs):
        """Update the display with emotion information"""
        h, w = frame.shape[:2]
        
        # Extract emotion predictions
        if 'level2_emotion_probs' in outputs:
            emotion_probs = outputs['level2_emotion_probs'][0, -1].cpu().numpy()
            
            # Get top emotion
            top_emotion_idx = np.argmax(emotion_probs)
            self.current_emotion = self.emotion_labels[top_emotion_idx]
            self.confidence = emotion_probs[top_emotion_idx]
            
            # Update history
            self.emotion_history.append({
                'emotion': self.current_emotion,
                'confidence': self.confidence,
                'probs': emotion_probs,
                'timestamp': time.time()
            })
        
        # Draw emotion label
        cv2.putText(frame, f"Emotion: {self.current_emotion}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {self.confidence:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw emotion probabilities bar chart
        bar_width = w // len(self.emotion_labels)
        bar_height = 100
        
        for i, (label, prob) in enumerate(zip(self.emotion_labels, emotion_probs)):
            x = i * bar_width
            height = int(prob * bar_height)
            color = self.emotion_colors[i]
            
            # Convert color name to BGR
            if color == 'red': bgr = (0, 0, 255)
            elif color == 'green': bgr = (0, 255, 0)
            elif color == 'blue': bgr = (255, 0, 0)
            elif color == 'yellow': bgr = (0, 255, 255)
            elif color == 'orange': bgr = (0, 165, 255)
            elif color == 'purple': bgr = (128, 0, 128)
            elif color == 'gray': bgr = (128, 128, 128)
            else: bgr = (42, 42, 165)  # brown
            
            cv2.rectangle(frame, (x, h - height), (x + bar_width - 2, h), bgr, -1)
            cv2.putText(frame, label[:3], (x + 5, h - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add hierarchical level indicators
        if 'level1_z1' in outputs:
            cv2.putText(frame, "L1: Active", (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        if 'level2_z2' in outputs:
            cv2.putText(frame, "L2: Active", (w - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        if 'level3_z3' in outputs:
            cv2.putText(frame, "L3: Active", (w - 150, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Add FPS counter
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_history.append(fps)
        avg_fps = np.mean(list(self.fps_history))
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (w - 150, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
        
    def draw_emotion_trajectory(self, frame):
        """Draw emotion trajectory visualization"""
        if len(self.emotion_history) < 2:
            return frame
            
        # Create trajectory plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
        
        # Emotion timeline
        timestamps = [e['timestamp'] for e in self.emotion_history]
        emotions = [e['emotion'] for e in self.emotion_history]
        confidences = [e['confidence'] for e in self.emotion_history]
        
        # Convert to relative time
        start_time = timestamps[0]
        rel_times = [(t - start_time) for t in timestamps]
        
        # Plot confidence over time
        ax1.plot(rel_times, confidences, 'b-', linewidth=2)
        ax1.set_ylabel('Confidence')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Plot emotion categories
        emotion_indices = [self.emotion_labels.index(e) for e in emotions]
        ax2.scatter(rel_times, emotion_indices, c=confidences, cmap='viridis', s=20)
        ax2.set_yticks(range(len(self.emotion_labels)))
        ax2.set_yticklabels(self.emotion_labels)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Emotion')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Resize and overlay on frame
        plot_resized = cv2.resize(plot_img, (400, 300))
        plot_bgr = cv2.cvtColor(plot_resized, cv2.COLOR_RGB2BGR)
        
        # Add semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 100), (410, 400), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add plot
        frame[100:400, 10:410] = plot_bgr
        
        return frame
        
    def run(self):
        """Main demo loop"""
        print("Starting real-time demo...")
        print("Press 'q' to quit, 's' to save screenshot, 't' to toggle trajectory")
        
        # Start audio stream
        self.start_audio_stream()
        
        show_trajectory = False
        frame_count = 0
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Add to buffer
                self.frame_buffer.append(frame)
                frame_count += 1
                
                # Process every 10 frames (3 times per second)
                if frame_count % 10 == 0:
                    outputs = self.process_hierarchical()
                    
                    if outputs is not None:
                        frame = self.update_emotion_display(frame, outputs)
                        
                        if show_trajectory:
                            frame = self.draw_emotion_trajectory(frame)
                
                # Display frame
                cv2.imshow('Hierarchical Emotional Intelligence Demo', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'demo_screenshot_{timestamp}.png', frame)
                    print(f"Screenshot saved as demo_screenshot_{timestamp}.png")
                elif key == ord('t'):
                    show_trajectory = not show_trajectory
                    
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            self.audio_stream.stop()
            self.audio_stream.close()
            

def main():
    parser = argparse.ArgumentParser(description='Real-time Hierarchical EI Demo')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model configuration')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found!")
        return
        
    if not Path(args.config).exists():
        print(f"Error: Config file {args.config} not found!")
        return
    
    # Create and run demo
    demo = RealTimeDemo(args.model, args.config, args.device)
    demo.run()


if __name__ == '__main__':
    main()
