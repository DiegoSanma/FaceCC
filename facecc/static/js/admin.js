let systemEnabled = true;

async function loadSystemStatus() {
    try {
        const resp = await fetch('/admin/system-status');
        const data = await resp.json();
        systemEnabled = data.recognition_enabled;
        document.getElementById('status-text').textContent = systemEnabled ? 'Activo' : 'Desactivado';
        document.getElementById('status-text').style.color = systemEnabled ? '#4CAF50' : '#f44336';
    } catch (e) {
        console.error('Error loading status:', e);
    }
}

document.getElementById('toggle-btn').addEventListener('click', async () => {
    try {
        const resp = await fetch('/admin/system-toggle', { method: 'POST' });
        const data = await resp.json();
        systemEnabled = data.recognition_enabled;
        document.getElementById('status-text').textContent = systemEnabled ? 'Activo' : 'Desactivado';
        document.getElementById('status-text').style.color = systemEnabled ? '#4CAF50' : '#f44336';
    } catch (e) {
        alert('Error toggling system');
    }
});

async function loadFaces() {
    try {
        const resp = await fetch('/admin/faces');
        const faces = await resp.json();
        const list = document.getElementById('faces-list');
        const count = document.getElementById('face-count');
        
        count.textContent = faces.length;
        list.innerHTML = '';
        
        faces.forEach(face => {
            const div = document.createElement('div');
            div.className = 'face-card';
            div.innerHTML = `
                <span class="face-name">${face.name}</span>
                <button class="delete-btn" data-id="${face.id}">Eliminar</button>
            `;
            list.appendChild(div);
        });
        
        document.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const id = e.target.dataset.id;
                if (confirm('¿Eliminar esta cara?')) {
                    await fetch(`/admin/faces/${id}`, { method: 'DELETE' });
                    loadFaces();
                }
            });
        });
    } catch (e) {
        console.error('Error loading faces:', e);
    }
}

document.getElementById('add-face-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('face-name').value.trim();
    const file = document.getElementById('face-image').files[0];
    const status = document.getElementById('upload-status');
    
    if (!name || !file) {
        status.textContent = 'Completa todos los campos';
        status.style.color = 'red';
        return;
    }
    
    const formData = new FormData();
    formData.append('name', name);
    formData.append('file', file);
    
    status.textContent = 'Subiendo...';
    status.style.color = 'blue';
    
    try {
        const resp = await fetch('/admin/faces', { method: 'POST', body: formData });
        const data = await resp.json();
        
        if (resp.ok) {
            status.textContent = 'Cara agregada correctamente';
            status.style.color = 'green';
            document.getElementById('add-face-form').reset();
            loadFaces();
        } else {
            status.textContent = `Error: ${data.error}`;
            status.style.color = 'red';
        }
    } catch (e) {
        status.textContent = 'Error de red';
        status.style.color = 'red';
    }
});

async function loadLogs() {
    try {
        const resp = await fetch('/admin/logs?lines=50');
        const logs = await resp.json();
        document.getElementById('logs-display').textContent = logs.join('\n') || 'Sin registros';
    } catch (e) {
        console.error('Error loading logs:', e);
    }
}

document.getElementById('refresh-logs').addEventListener('click', loadLogs);

// Init
loadSystemStatus();
loadFaces();
loadLogs();
setInterval(loadLogs, 10000); // auto-refresh logs every 10s