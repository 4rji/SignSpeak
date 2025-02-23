import * as handpose from '@tensorflow-models/handpose';
import '@tensorflow/tfjs-backend-webgl';

let handposeModel;
let video;
let predictions = [];
let word = [];
let lastLetter = "";
let staticGestureCount = 0;

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20]
];

async function init() {
  try {
    video = document.getElementById('videoElement');
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: 640, height: 480 }
    });
    video.srcObject = stream;
    await new Promise(resolve => (video.onloadedmetadata = resolve));
    video.play();

    handposeModel = await handpose.load();
    console.log("Handpose model loaded");
    
    detectFrame();
  } catch (error) {
    console.error("Error during initialization:", error);
    document.getElementById('result').textContent = `Error: ${error.message}`;
  }
}

async function detectFrame() {
  try {
    const canvas = document.getElementById('canvasElement');
    const ctx = canvas.getContext('2d');
    predictions = await handposeModel.estimateHands(video);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (predictions.length > 0) {
      drawHand(predictions[0], ctx);
      const landmarks = predictions[0].landmarks;
      const predictedLetter = interpretGesture(landmarks);
      
      // Add letter detection logic
      if (predictedLetter !== lastLetter) {
        staticGestureCount = 0;
        lastLetter = predictedLetter;
      } else {
        staticGestureCount++;
        if (staticGestureCount > 30) { // About 1 second at 30fps
          if (!word.includes(predictedLetter)) {
            word.push(predictedLetter);
            staticGestureCount = 0;
          }
        }
      }
      
      document.getElementById('result').textContent = `Detected letter: ${predictedLetter || 'None'}\nWord: ${word.join('')}`;
    } else {
      document.getElementById('result').textContent = 'No hands detected';
    }
    requestAnimationFrame(detectFrame);
  } catch (error) {
    console.error("Detection error:", error);
    document.getElementById('result').textContent = `Detection error: ${error.message}`;
  }
}

function drawHand(prediction, ctx) {
  prediction.landmarks.forEach(point => {
    ctx.beginPath();
    ctx.arc(point[0], point[1], 5, 0, 2 * Math.PI);
    ctx.fillStyle = 'red';
    ctx.fill();
  });
  HAND_CONNECTIONS.forEach(([start, end]) => {
    ctx.beginPath();
    ctx.moveTo(prediction.landmarks[start][0], prediction.landmarks[start][1]);
    ctx.lineTo(prediction.landmarks[end][0], prediction.landmarks[end][1]);
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}

function interpretGesture(landmarks) {
  // Here's a basic implementation for the letter 'A'
  // You'll need to add more gesture recognition logic for other letters
  
  // Get thumb tip and index finger tip
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  
  // Calculate distance between thumb and index finger
  const distance = Math.sqrt(
    Math.pow(thumbTip[0] - indexTip[0], 2) + 
    Math.pow(thumbTip[1] - indexTip[1], 2)
  );
  
  // If thumb and index finger are close together (forming an 'A')
  if (distance < 30) {
    return 'A';
  }
  
  return null;
}

export function clearText() {
  word = [];
  document.getElementById('result').textContent = '';
}

window.addEventListener('load', init);
