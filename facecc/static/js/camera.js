const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const context = canvas.getContext("2d");
const button = document.getElementById("takePhotoBtn");
const resultDiv = document.getElementById("result");
detectionInterval = null;

let faceDetected = false;
let detectionStart = null; // timestamp de cuando apareció la cara
let photosent = false;

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
    const MODEL_URL = 'facecc/static/models';
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
}

// Detectar caras en tiempo real
async function detectFaces() {
    const options = new faceapi.TinyFaceDetectorOptions();

    detectionInterval =  setInterval(async () => {
        const detections = await faceapi.detectAllFaces(video, options);
        context.clearRect(0, 0, canvas.width, canvas.height);

        if (detections.length > 0) {
            // Dibuja el cuadro alrededor de la cara
            const { x, y, width, height } = detections[0].box;
            context.strokeStyle = "#ff0000ff";
            context.lineWidth = 2;
            context.strokeRect(x, y, width, height);

            if (!faceDetected) {
                faceDetected = true;
                detectionStart = Date.now(); // inicio del conteo
            } else {
                const elapsed = Date.now() - detectionStart;
                if (elapsed >= 3000 && !photosent) { // 3 segundos
                    captureFrame(); // capturar foto
                    faceDetected = false; // reiniciar contador
                    photosent = true;
                }
            }
        } else {
            // Si no hay cara, reinicia
            faceDetected = false;
            detectionStart = null;
        }
    }, 100); // cada 100ms
}

async function captureFrame() {
    const canvasCapture = document.createElement("canvas");
    canvasCapture.width = video.videoWidth;
    canvasCapture.height = video.videoHeight;
    const ctx = canvasCapture.getContext("2d");
    ctx.drawImage(video, 0, 0, canvasCapture.width, canvasCapture.height);
    const imageBase64 = canvasCapture.toDataURL("image/jpeg");

    const response = await fetch("facecc/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageBase64, csrfmiddlewaretoken: getCookie('csrftoken') }),
    });
    const result = await response.json();
    showresult(result.name);
};

function showresult(name){
    if(detectionInterval){
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
    if(name=="Desconocido"){
        context.strokeStyle = "#ff0000ff"
        context.lineWidth = 4;
        resultDiv.innerHTML = "No reconocido, intente de nuevo";
        resultDiv.style.display = "block";
        setTimeout(() => {
            resultDiv.style.display = "none";
            button.style.display = "block"; // Mostrar el botón nuevamente
            video.srcObject.getTracks().forEach(track => track.stop()); // Detener la cámara
            faceDetected = false; // Reiniciar el estado de detección
            detectionStart = null; // Reiniciar el timestamp
            context.clearRect(0, 0, overlay.width, overlay.height);
            photosent = false;
        }, 3000);
        return;
    }
    else{
        resultDiv.innerHTML = `Hola, ${name}`;
        resultDiv.style.display = "block";
        context.strokeStyle = "#1ee614ff"
        context.lineWidth = 4;
        setTimeout(() => {
            resultDiv.style.display = "none";
            resultDiv.innerHTML = `Esperando`
            button.style.display = "block"; // Mostrar el botón nuevamente
            video.srcObject.getTracks().forEach(track => track.stop()); // Detener la cámara
            faceDetected = false; // Reiniciar el estado de detección
            detectionStart = null; // Reiniciar el timestamp
            context.clearRect(0, 0, overlay.width, overlay.height);
            photosent = false;
        }, 5000);
    }
}

// Iniciar todo
button.addEventListener("click", async () => {
    button.style.display = "none"; // Ocultar el botón después de hacer clic
    resultDiv.style.display = "block"; // Ocultar el resultado si está visible
    await loadModels();
    await startCamera();
    detectFaces();
});

