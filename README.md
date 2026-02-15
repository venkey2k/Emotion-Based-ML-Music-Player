# Moodify — Emotion Based ML Music Player

Moodify is a modern music player that uses machine learning to detect your current emotion via webcam and plays music that matches your mood.

## Features

- **Real-time Emotion Detection**: Uses a CNN model with v7 accuracy fixes including TTA, CLAHE, and Geometric boosting.
- **Mood-based Playlists**: Automatically categorizes and plays songs based on detected emotions (Angry, Happy, Neutral, Sad, Surprise).
- **YouTube Integration**: Download and tag songs directly from YouTube into mood folders.
- **Modern UI**: Built with Eel (Python + HTML/JS/CSS) for a responsive and beautiful desktop experience.
- **Library Watcher**: Automatically detects new songs added to the `songs/` directory.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Emotion Based ML Music Player"
   ```

2. **Set up virtual environment (recommended)**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the application**:
   ```bash
   python main.py
   ```
2. **Calibrate**: For best results, show a neutral face and click 'Calibrate' (or press 'c' in live detector).
3. **Enjoy**: The player will suggest songs based on your detected mood.

## Project Structure

- `main.py`: Main application entry point using Eel.
- `emotion_detector.py`: Core logic for emotion detection (CNN + Geometric).
- `live_detector.py`: Standalone live detection tool for testing.
- `watcher.py`: File system watcher for the music library.
- `youtube_dl_module.py`: Handles YouTube audio downloads.
- `songs/`: Directory where music is stored by mood.
- `keras/`: Contains the pre-trained models and cascades.

## Controls (Live Detector)
- `q`: Quit
- `s`: Screenshot
- `c`: Calibrate neutral
- `t`: Toggle TTA
- `e`: Toggle CLAHE
- `g`: Toggle Geo Boost
- `b`: Toggle Class Boost
