// Obtener referencias al video y al botón
const video = document.getElementById("camera");
const button = document.getElementById("snap");

// Función para iniciar la cámara
async function startCamera() {
    console.log("Iniciando cámara...");
    try {
        // Solicitar acceso a la cámara
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;   // Mostrar la cámara en el <video>
    } catch (err) {
        alert("No se pudo acceder a la cámara. Verifica los permisos.");
        console.error(err);
    }
}

// Función para capturar un frame y enviarlo a un formulario (opcional)
function captureFrame() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg");  // Devuelve la imagen en base64
}

// Ejemplo: acción del botón
button.addEventListener("click", () => {
    const frame = captureFrame();
    console.log("Frame capturado:", frame); // Aquí puedes enviarlo al backend
    alert("Foto tomada! (revisa la consola para el base64)");
});

// Iniciar la cámara cuando cargue la página
window.addEventListener("load", startCamera);

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
