// Importing required modules
const { app, BrowserWindow } = require('electron')

function createWindow () {
  // Create the browser window.
  let win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
    }
  })

  // Load an HTML file into the window.
  win.loadFile('index.html')
}

app.whenReady().then(createWindow)
