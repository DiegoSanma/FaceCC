const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const context = canvas.getContext("2d");

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

        detections.forEach(det => {
            const { x, y, width, height } = det.box;
            context.strokeStyle = "#00FFFF";
            context.lineWidth = 2;
            context.strokeRect(x, y, width, height);
        });
    }, 100); // cada 100ms
}

// Iniciar todo
window.addEventListener("load", async () => {
    await loadModels();
    await startCamera();
    detectFaces();
});


const snap = document.getElementById("snap");
const imageInput = document.getElementById("imageInput");
const form = document.getElementById("photoForm")
navigator.mediaDevices.getUserMedia({video:true}).then(stream => video.srcObject = stream)
snap.addEventListener("click", () => {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    imageInput.value = canvas.toDataURL("image/png");
    form.submit();
});
