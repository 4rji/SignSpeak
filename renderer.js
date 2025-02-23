const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const output = document.getElementById('output');
const words = document.getElementById('words');
const clearButton = document.getElementById('clear');
let hands;
let camera;

const TOL = 0.12;
const FINGER_SPREAD_TOL = 0.15;
const LETTER_TIMEOUT = 1500;

let isDetecting = true; // Variable para controlar la detección

// Definición de gestos para cada letra del alfabeto (A-Z)
const ASL_GESTURES = {
    'A': (landmarks) => {
        const palm = landmarks[0];
        const thumb = landmarks[4];
        const index = landmarks[8];
        const middle = landmarks[12];
        const ring = landmarks[16];
        const pinky = landmarks[20];
        return index.y > palm.y - TOL &&
               middle.y > palm.y - TOL &&
               ring.y > palm.y - TOL &&
               pinky.y > palm.y - TOL &&
               thumb.x > palm.x;
    },
    'B': (landmarks) => {
        const palm = landmarks[0];
        const thumb = landmarks[4];
        const index = landmarks[8];
        const middle = landmarks[12];
        const ring = landmarks[16];
        const pinky = landmarks[20];
        return index.y < palm.y - TOL &&
               middle.y < palm.y - TOL &&
               ring.y < palm.y - TOL &&
               pinky.y < palm.y - TOL &&
               thumb.x < palm.x &&
               Math.abs(index.x - middle.x) < TOL &&
               Math.abs(middle.x - ring.x) < TOL &&
               Math.abs(ring.x - pinky.x) < TOL;
    },
    'C': (landmarks) => {
        const thumb = landmarks[4];
        const index = landmarks[8];
        const middle = landmarks[12];
        const ring = landmarks[16];
        const pinky = landmarks[20];
        return Math.abs(thumb.x - pinky.x) < 0.3 &&
               thumb.y > middle.y &&
               pinky.y > middle.y &&
               index.y < middle.y;
    },
    'D': (landmarks) => {
        const palm = landmarks[0];
        const thumb = landmarks[4];
        const index = landmarks[8];
        const middle = landmarks[12];
        return index.y < palm.y - TOL &&
               middle.y > palm.y - TOL &&
               Math.abs(thumb.x - middle.x) < TOL;
    },
    'E': (landmarks) => {
        const palm = landmarks[0];
        const thumb = landmarks[4];
        const fingers = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]];
        return fingers.every(finger => Math.abs(finger.y - palm.y) < TOL) &&
               thumb.x > palm.x;
    },
    'F': (landmarks) => {
        const palm = landmarks[0];
        const thumb = landmarks[4];
        const index = landmarks[8];
        const middle = landmarks[12];
        return Math.abs(thumb.y - index.y) < TOL &&
               middle.y < palm.y &&
               thumb.x < index.x;
    },
    'G': (landmarks) => {
        const palm = landmarks[0];
        const index = landmarks[8];
        const middle = landmarks[12];
        return index.x > palm.x &&
               index.y < palm.y &&
               middle.y > palm.y;
    },
    'H': (landmarks) => {
        const palm = landmarks[0];
        const index = landmarks[8];
        const middle = landmarks[12];
        return index.y < palm.y &&
               middle.y < palm.y &&
               index.x > middle.x;
    },
    'I': (landmarks) => {
        const palm = landmarks[0];
        const pinky = landmarks[20];
        const ring = landmarks[16];
        return pinky.y < palm.y - TOL &&
               ring.y > palm.y - TOL;
    },
    'J': (landmarks) => {
        const palm = landmarks[0];
        const pinky = landmarks[20];
        return pinky.y < palm.y &&
               pinky.x < palm.x;
    },
    'K': (landmarks) => {
        const palm = landmarks[0];
        const index = landmarks[8];
        const middle = landmarks[12];
        return index.y < palm.y &&
               middle.y < palm.y &&
               index.x < middle.x;
    },
    'L': (landmarks) => {
        const palm = landmarks[0];
        const thumb = landmarks[4];
        const index = landmarks[8];
        return thumb.x < palm.x &&
               index.y < palm.y &&
               Math.abs(thumb.x - index.x) > TOL;
    },
    'M': (landmarks) => {
        const palm = landmarks[0];
        const index = landmarks[8];
        const middle = landmarks[12];
        const ring = landmarks[16];
        return index.y < palm.y &&
               middle.y < palm.y &&
               ring.y < palm.y &&
               Math.abs(index.x - middle.x) < TOL &&
               Math.abs(middle.x - ring.x) < TOL;
    },
    'N': (landmarks) => {
        const palm = landmarks[0];
        const index = landmarks[8];
        const middle = landmarks[12];
        return index.y < palm.y &&
               middle.y < palm.y &&
               Math.abs(index.x - middle.x) < TOL;
    },
    'O': (landmarks) => {
        const thumb = landmarks[4];
        const index = landmarks[8];
        return Math.abs(thumb.x - index.x) < TOL &&
               Math.abs(thumb.y - index.y) < TOL;
    },
    'P': (landmarks) => {
        const palm = landmarks[0];
        const thumb = landmarks[4];
        const index = landmarks[8];
        return index.y > palm.y &&
               thumb.x < index.x;
    },
    'Q': (landmarks) => {
        const palm = landmarks[0];
        const thumb = landmarks[4];
        const index = landmarks[8];
        return index.y > palm.y &&
               thumb.x > index.x;
    },
    'R': (landmarks) => {
        const palm = landmarks[0];
        const index = landmarks[8];
        const middle = landmarks[12];
        return index.y < palm.y &&
               middle.y < palm.y &&
               Math.abs(index.x - middle.x) < TOL &&
               index.x < middle.x;
    },
    'S': (landmarks) => {
        const palm = landmarks[0];
        const thumb = landmarks[4];
        const index = landmarks[8];
        return Math.abs(thumb.y - palm.y) < TOL &&
               Math.abs(index.y - palm.y) < TOL;
    },
    'T': (landmarks) => {
        const palm = landmarks[0];
        const thumb = landmarks[4];
        const index = landmarks[8];
        return thumb.y < palm.y &&
               Math.abs(index.y - palm.y) < TOL;
    },
    'U': (landmarks) => {
        const palm = landmarks[0];
        const index = landmarks[8];
        const middle = landmarks[12];
        return index.y < palm.y &&
               middle.y < palm.y &&
               Math.abs(index.x - middle.x) < TOL;
    },
    'V': (landmarks) => {
        const palm = landmarks[0];
        const index = landmarks[8];
        const middle = landmarks[12];
        const ring = landmarks[16];
        const pinky = landmarks[20];
        
        const fingersExtended = index.y < palm.y - TOL && 
                               middle.y < palm.y - TOL;
        
        const fingersDown = ring.y > palm.y - TOL && 
                           pinky.y > palm.y - TOL;
        
        const properSpread = Math.abs(index.x - middle.x) > FINGER_SPREAD_TOL;
        
        return fingersExtended && fingersDown && properSpread;
    },
    'W': (landmarks) => {
        const palm = landmarks[0];
        const index = landmarks[8];
        const middle = landmarks[12];
        const ring = landmarks[16];
        return index.y < palm.y &&
               middle.y < palm.y &&
               ring.y < palm.y &&
               Math.abs(index.x - middle.x) > TOL &&
               Math.abs(middle.x - ring.x) > TOL;
    },
    'X': (landmarks) => {
        const palm = landmarks[0];
        const index = landmarks[8];
        return index.y < palm.y &&
               index.x > palm.x;
    },
    'Y': (landmarks) => {
        const palm = landmarks[0];
        const thumb = landmarks[4];
        const pinky = landmarks[20];
        return thumb.y < palm.y &&
               pinky.y < palm.y &&
               Math.abs(thumb.x - pinky.x) > TOL;
    },
    'Z': (landmarks) => {
        const palm = landmarks[0];
        const index = landmarks[8];
        return index.y < palm.y &&
               index.x > palm.x;
    }
};

// Orden de letras para detección
const KNOWN_WORDS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
];

// Buffers y tiempos para cada mano
const letterBuffers = {};
const lastDetectedTimes = {};

// Se recorre el alfabeto en orden y se retorna la primera letra cuyo gesto se cumpla
function interpretGesture(landmarks) {
    for (const letter of KNOWN_WORDS) {
        if (ASL_GESTURES[letter] && ASL_GESTURES[letter](landmarks)) {
            return letter;
        }
    }
    return null;
}

function onResults(results) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        results.multiHandLandmarks.forEach((landmarks, index) => {
            const letter = interpretGesture(landmarks);
            const color = letter ? '#00FF00' : '#FF0000';
            drawHand(landmarks, color);

            if (letter && isDetecting) {
                const currentTime = Date.now();
                
                if (!lastDetectedTimes[index] || 
                    currentTime - lastDetectedTimes[index] > LETTER_TIMEOUT) {
                    
                    if (currentTime - lastDetectedTimes[index] > LETTER_TIMEOUT * 2) {
                        letterBuffers[index] = '';
                    }
                    
                    if (!letterBuffers[index] || 
                        letterBuffers[index].charAt(letterBuffers[index].length - 1) !== letter) {
                        letterBuffers[index] = (letterBuffers[index] || '') + letter;
                    }
                }
                
                lastDetectedTimes[index] = currentTime;
                isDetecting = false;
                
                output.textContent = `Letra actual: ${letter} | Buffer: ${letterBuffers[index]}`;
                output.style.color = '#000000';
                
                setTimeout(() => {
                    isDetecting = true;
                }, 500);
            }
        });
    } else {
        output.textContent = 'Esperando detección de manos...';
        output.style.color = '#666666';
    }
}

function drawHand(landmarks, color = 'blue') {
    // Dibuja los puntos
    for (const landmark of landmarks) {
        const x = landmark.x * canvas.width;
        const y = landmark.y * canvas.height;
        context.beginPath();
        context.arc(x, y, 5, 0, 2 * Math.PI);
        context.fillStyle = color;
        context.fill();
        context.beginPath();
        context.arc(x, y, 7, 0, 2 * Math.PI);
        context.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        context.stroke();
    }
    // Dibuja las conexiones
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20]
    ];
    context.beginPath();
    for (const [start, end] of connections) {
        const startX = landmarks[start].x * canvas.width;
        const startY = landmarks[start].y * canvas.height;
        const endX = landmarks[end].x * canvas.width;
        const endY = landmarks[end].y * canvas.height;
        context.moveTo(startX, startY);
        context.lineTo(endX, endY);
    }
    context.strokeStyle = color;
    context.lineWidth = 3;
    context.stroke();
}

async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;
        return new Promise(resolve => {
            video.onloadedmetadata = () => {
                video.play();
                output.textContent = "Cámara iniciada";
                resolve(video);
            };
        });
    } catch (error) {
        output.textContent = "Error al acceder a la cámara: " + error.message;
        console.error('Error al acceder a la cámara:', error);
    }
}

async function init() {
    try {
        hands = new Hands({
            locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });
        hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.7,
            selfieMode: true
        });
        hands.onResults(onResults);
        camera = new Camera(video, {
            onFrame: async () => { await hands.send({ image: video }); },
            width: 640,
            height: 480
        });
        output.textContent = 'Starting camera...';
        await camera.start();
        output.textContent = 'Sistema listo. Detección iniciada...';
    } catch (error) {
        console.error('Error during initialization:', error);
        output.textContent = 'Error al iniciar: ' + error.message;
    }
}

async function startApp() {
    try {
        await setupCamera();
        await init();
    } catch (error) {
        output.textContent = "Error al iniciar la aplicación: " + error.message;
        console.error('Error al iniciar la aplicación:', error);
    }
}

clearButton.addEventListener('click', () => {
    words.textContent = '';
    Object.keys(letterBuffers).forEach(key => delete letterBuffers[key]);
    Object.keys(lastDetectedTimes).forEach(key => delete lastDetectedTimes[key]);
    output.textContent = 'Sistema listo para detectar...';
});

startApp();
