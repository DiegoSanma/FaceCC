const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const context = canvas.getContext("2d");

let faceDetected = false;
let detectionStart = null; // timestamp de cuando apareció la cara

//Obtengo la cookie del csrftoken
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}


async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        alert("No se pudo acceder a la cámara. Verifica los permisos.");
        console.error(err);
    }
}

// Cargar los modelos de face-api.js
async function loadModels() {
    const MODEL_URL = '/static/models';
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
}

// Detectar caras en tiempo real
async function detectFaces() {
    const options = new faceapi.TinyFaceDetectorOptions();

    setInterval(async () => {
        const detections = await faceapi.detectAllFaces(video, options);
        context.clearRect(0, 0, canvas.width, canvas.height);

        if (detections.length > 0) {
            // Dibuja el cuadro alrededor de la cara
            const { x, y, width, height } = detections[0].box;
            context.strokeStyle = "#00FFFF";
            context.lineWidth = 2;
            context.strokeRect(x, y, width, height);

            if (!faceDetected) {
                faceDetected = true;
                detectionStart = Date.now(); // inicio del conteo
            } else {
                const elapsed = Date.now() - detectionStart;
                if (elapsed >= 3000) { // 3 segundos
                    captureFrame(); // capturar foto
                    faceDetected = false; // reiniciar contador
                }
            }
        } else {
            // Si no hay cara, reinicia
            faceDetected = false;
            detectionStart = null;
        }
    }, 100); // cada 100ms
}

function captureFrame() {
    const canvasCapture = document.createElement("canvas");
    canvasCapture.width = video.videoWidth;
    canvasCapture.height = video.videoHeight;
    const ctx = canvasCapture.getContext("2d");
    ctx.drawImage(video, 0, 0, canvasCapture.width, canvasCapture.height);
    const imageBase64 = canvasCapture.toDataURL("image/jpeg");

    console.log("Foto capturada:", imageBase64);

    fetch('/process_image/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken') // Asegúrate de tener la función getCookie definida
        },
        body: JSON.stringify({ image: imageBase64 })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Respuesta del servidor:', data);
        alert(data.message);
    })
    .catch(error => {
        console.error('Error al enviar la imagen:', error);
    });
    
}


// Iniciar todo
window.addEventListener("load", async () => {
    await loadModels();
    await startCamera();
    detectFaces();
});

