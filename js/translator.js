let handpose;
let video;
let predictions = [];
let word = [];
let lastLetter = "";
let staticGestureCount = 0;

// Cargar el modelo de clasificación de gestos
const gestureModelPath = 'models/gesture_model/model.json';
let gestureClassifier;

// Conexiones entre puntos de la mano para dibujar
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], // pulgar
    [0, 5], [5, 6], [6, 7], [7, 8], // índice
    [0, 9], [9, 10], [10, 11], [11, 12], // medio
    [0, 13], [13, 14], [14, 15], [15, 16], // anular
    [0, 17], [17, 18], [18, 19], [19, 20] // meñique
];

// Diccionario de letras ASL
const ASL_LETTERS = {
    'A': [/* configuración de puntos para A */],
    'B': [/* configuración de puntos para B */],
    // ... más letras
};

async function init() {
    try {
        // Configurar video
        video = document.getElementById('videoElement');
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: {
                width: 640,
                height: 480
            }
        });
        video.srcObject = stream;
        
        // Esperar a que el video esté listo
        await new Promise(resolve => video.onloadedmetadata = resolve);
        video.play();

        // Crear canvas para dibujar
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        document.querySelector('.container').insertBefore(canvas, document.getElementById('result'));
        const ctx = canvas.getContext('2d');

        // Cargar modelo Handpose - Corregido
        handpose = await handpose.load();
        console.log("Modelo Handpose cargado");

        // Comenzar la detección
        detectFrame(ctx);
    } catch (error) {
        console.error("Error al inicializar:", error);
        document.getElementById('result').textContent = 
            `Error: ${error.message}. Por favor, asegúrate de dar permisos de cámara y recargar la página.`;
    }
}

async function detectFrame(ctx) {
    try {
        // Detectar manos
        predictions = await handpose.estimate(video);

        // Limpiar canvas
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.drawImage(video, 0, 0, ctx.canvas.width, ctx.canvas.height);

        if (predictions.length > 0) {
            // Dibujar puntos y conexiones
            drawHand(predictions[0], ctx);
            
            // Interpretar gesto
            const landmarks = predictions[0].landmarks;
            const predictedLetter = interpretGesture(landmarks);
            
            // Mostrar letra detectada en tiempo real
            document.getElementById('result').textContent = 
                `Letra detectada: ${predictedLetter || 'Ninguna'}`;
        } else {
            document.getElementById('result').textContent = 'No se detectan manos';
        }

        // Continuar la detección
        requestAnimationFrame(() => detectFrame(ctx));
    } catch (error) {
        console.error("Error en detección:", error);
        document.getElementById('result').textContent = `Error en detección: ${error.message}`;
    }
}

function drawHand(prediction, ctx) {
    // Dibujar puntos
    prediction.landmarks.forEach(point => {
        ctx.beginPath();
        ctx.arc(point[0], point[1], 5, 0, 2 * Math.PI);
        ctx.fillStyle = 'red';
        ctx.fill();
    });

    // Dibujar conexiones
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
    // Por ahora, solo devolvemos una letra de prueba
    return 'A';
}

function handleStaticGesture(predictedLetter) {
    if (predictedLetter === lastLetter) {
        staticGestureCount++;
        if (staticGestureCount > 10) { // Ajustar según necesidad
            word.push(predictedLetter);
            staticGestureCount = 0;
        }
    } else {
        lastLetter = predictedLetter;
        staticGestureCount = 0;
    }
}

function handleNoHandDetected() {
    if (word.length > 0 && word[word.length - 1] !== " ") {
        staticGestureCount++;
        if (staticGestureCount > 10) {
            word.push(" ");
            staticGestureCount = 0;
        }
    }
}

function updateResult() {
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = word.join("");
}

// Función para limpiar el texto
function clearText() {
    word = [];
    document.getElementById('result').textContent = '';
}

// Iniciar cuando la página esté cargada
window.addEventListener('load', init); 