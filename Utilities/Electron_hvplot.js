// Importing required modules
const { app, BrowserWindow } = require('electron')
app.disableHardwareAcceleration();

function createWindow () {
  let win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      sandbox: false,
    }
  })

  // Load your plot HTML file
  win.loadFile('/Users/ulric/Documents/Hvplots/SignificantEvents_20240131_1544.html')
}
app.whenReady().then(createWindow)