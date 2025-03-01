# SignSpeak - ASL Alphabet Translator

SignSpeak is a real-time American Sign Language (ASL) alphabet translator that uses computer vision to detect and interpret hand gestures through your webcam. This web application helps users learn and practice the ASL alphabet by providing immediate visual feedback.

![SignSpeak Demo](demo.gif)

## Features

- Real-time hand gesture detection
- ASL alphabet recognition (A-Z)
- Visual feedback with hand landmark tracking
- Interactive gesture guide
- Clean and intuitive user interface
- Buffer system to prevent accidental detections
- Clear function to reset detected letters

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Computer Vision**: MediaPipe Hands API
- **Machine Learning**: TensorFlow.js
- **Video Processing**: HTML5 Canvas
- **Styling**: Custom CSS with CSS Variables

## How It Works

1. The application accesses your webcam feed
2. MediaPipe Hands API tracks hand landmarks in real-time
3. Custom algorithms compare hand positions against predefined ASL letter patterns
4. Visual feedback is provided through:
   - Green markers when a letter is recognized
   - Red markers when no letter is detected
   - Text display showing current detected letter
   - Buffer display showing sequence of detected letters

## ASL Alphabet Guide

The application includes a comprehensive guide for all 26 letters of the ASL alphabet, with descriptions of how to form each letter:

- A: Closed fist with thumb on side
- B: Flat hand, fingers together
- C: Curved hand in C shape
- ...and so on for all letters

## Getting Started

1. Clone the repository
2. Open `index.html` in a modern web browser
3. Allow camera access when prompted
4. Position your hand in front of the camera
5. Practice ASL letters following the guide

## Requirements

- Modern web browser with JavaScript enabled
- Webcam
- Internet connection (for loading required libraries)

## Privacy

The application processes all video locally in your browser. No video data is transmitted or stored.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for their excellent hand tracking solution
- TensorFlow.js team for their machine learning framework
- The ASL community for inspiration and guidance

## Support

If you encounter any issues or have questions, please open an issue in the GitHub repository.
