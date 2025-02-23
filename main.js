const { app, BrowserWindow } = require('electron')
const path = require('path')

app.disableHardwareAcceleration()

function createWindow () {
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    frame: true,
    show: true,
    backgroundColor: '#ffffff'
  })

  mainWindow.loadFile('index.html')
  
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('Failed to load:', errorDescription)
  })

  mainWindow.webContents.openDevTools()
}

app.on('ready', () => {
  try {
    createWindow()
  } catch (error) {
    console.error('Error creating window:', error)
    app.quit()
  }
})

app.on('window-all-closed', function () {
  app.quit()
})

app.on('activate', function () {
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})

process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error)
  app.quit()
}) 