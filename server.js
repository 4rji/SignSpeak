const express = require('express');
const path = require('path');

const app = express();
const port = 8501;

// Configurar el servidor para archivos estáticos
app.use(express.static(__dirname));

// Ruta principal
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Iniciar el servidor
app.listen(port, () => {
    console.log(`Servidor web iniciado en http://localhost:${port}`);
}); 