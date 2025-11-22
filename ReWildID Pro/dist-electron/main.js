"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const path_1 = __importDefault(require("path"));
const controller_1 = require("./controller");
// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
    electron_1.app.quit();
}
let mainWindow = null;
function createWindow() {
    mainWindow = new electron_1.BrowserWindow({
        width: 1280,
        height: 800,
        frame: false, // Make the window frameless
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path_1.default.join(__dirname, 'preload.js'),
            webSecurity: false, // Keeping consistent with neurolink/care-electron
            allowRunningInsecureContent: false,
        },
    });
    // Grant media permissions (covers webcam, microphone, and screen recording)
    mainWindow.webContents.session.setPermissionRequestHandler((webContents, permission, callback) => {
        if (permission === 'media') {
            callback(true);
        }
        else {
            callback(false);
        }
    });
    // Handle permission checks
    mainWindow.webContents.session.setPermissionCheckHandler((webContents, permission) => {
        if (permission === 'media') {
            return true;
        }
        return false;
    });
    // Set up display media request handler for screen sharing
    mainWindow.webContents.session.setDisplayMediaRequestHandler((request, callback) => {
        electron_1.desktopCapturer.getSources({ types: ['screen', 'window'] }).then((sources) => {
            const screenSource = sources.find(source => source.id.startsWith('screen:')) || sources[0];
            if (screenSource) {
                callback({ video: screenSource, audio: 'loopback' });
            }
            else {
                callback({});
            }
        }).catch(error => {
            console.error('Failed to get desktop capture sources:', error);
            callback({});
        });
    }, { useSystemPicker: true });
    // Load the app
    if (process.env.VITE_DEV_SERVER_URL) {
        mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
        mainWindow.webContents.openDevTools();
    }
    else {
        mainWindow.loadFile(path_1.default.join(__dirname, '../dist/index.html'));
    }
    // Open external links in default browser
    mainWindow.webContents.setWindowOpenHandler((details) => {
        electron_1.shell.openExternal(details.url);
        return { action: 'deny' };
    });
    // Add event listeners to track window state changes
    mainWindow.on('maximize', () => {
        mainWindow?.webContents.send('window-state-changed', { isMaximized: true });
    });
    mainWindow.on('unmaximize', () => {
        mainWindow?.webContents.send('window-state-changed', { isMaximized: false });
    });
    mainWindow.on('restore', () => {
        mainWindow?.webContents.send('window-state-changed', { isMaximized: false });
    });
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}
// Stream function for AI output
const stream = (txt) => {
    if (mainWindow) {
        mainWindow.webContents.send('stream', txt);
    }
    else {
        console.log('null mainWindow, cannot send stream data');
    }
};
// IPC handlers for window controls
electron_1.ipcMain.handle('window:minimize', () => {
    if (mainWindow)
        mainWindow.minimize();
});
electron_1.ipcMain.handle('window:maximize', () => {
    if (mainWindow) {
        if (mainWindow.isMaximized()) {
            mainWindow.unmaximize();
        }
        else {
            mainWindow.maximize();
        }
    }
});
electron_1.ipcMain.handle('window:close', () => {
    if (mainWindow)
        mainWindow.close();
});
electron_1.ipcMain.handle('window:isMaximized', () => {
    return mainWindow ? mainWindow.isMaximized() : false;
});
// IPC handlers for backend logic
electron_1.ipcMain.handle('browseImage', (_, date, folderPath) => (0, controller_1.browseImage)(date, folderPath));
electron_1.ipcMain.handle('viewImage', (_, originalPath) => (0, controller_1.viewImage)(originalPath));
electron_1.ipcMain.handle('getImagePaths', (_, currentFolder) => (0, controller_1.getImagePaths)(currentFolder));
electron_1.ipcMain.handle('getImages', (_, filter) => (0, controller_1.getImages)(filter));
electron_1.ipcMain.handle('downloadSelectedGalleryImages', (_, selectedPaths) => (0, controller_1.downloadSelectedGalleryImages)(selectedPaths));
electron_1.ipcMain.handle('uploadImage', (_, relativePath, originalPath) => (0, controller_1.uploadImage)(relativePath, originalPath));
electron_1.ipcMain.handle('uploadPaths', (_, filePaths, groupName) => (0, controller_1.uploadPaths)(filePaths, groupName));
electron_1.ipcMain.handle('deleteGroup', (_, id) => (0, controller_1.deleteGroup)(id));
electron_1.ipcMain.handle('deleteImage', (_, id) => (0, controller_1.deleteImage)(id));
electron_1.ipcMain.handle('updateGroupName', (_, id, name) => (0, controller_1.updateGroupName)(id, name));
electron_1.ipcMain.handle('checkIsDirectory', (_, filePath) => (0, controller_1.checkIsDirectory)(filePath));
electron_1.ipcMain.handle('detect', (_, selectedPaths) => (0, controller_1.detect)(selectedPaths, stream));
electron_1.ipcMain.handle('browseDetectImage', (_, date, folderPath, filterLabel, confLow, confHigh) => (0, controller_1.browseDetectImage)(date, folderPath, filterLabel, confLow, confHigh));
electron_1.ipcMain.handle('viewDetectImage', (_, date, imagePath) => (0, controller_1.viewDetectImage)(date, imagePath));
electron_1.ipcMain.handle('getDetectImagePaths', (_, dirPath, filterLabel, confLow, confHigh) => (0, controller_1.getDetectImagePaths)(dirPath, filterLabel, confLow, confHigh));
electron_1.ipcMain.handle('downloadDetectImages', (_, filterLabel) => (0, controller_1.downloadDetectImages)(filterLabel));
electron_1.ipcMain.handle('downloadSelectedDetectImages', (_, selectPaths) => (0, controller_1.downloadSelectedDetectImages)(selectPaths));
electron_1.ipcMain.handle('runReid', (_, selectedPaths) => (0, controller_1.runReid)(selectedPaths, stream));
electron_1.ipcMain.handle('browseReidImage', (_, date, time, group_id) => (0, controller_1.browseReidImage)(date, time, group_id));
electron_1.ipcMain.handle('downloadReidImages', (_, date, time) => (0, controller_1.downloadReidImages)(date, time));
electron_1.ipcMain.handle('deleteReidResult', (_, date, time) => (0, controller_1.deleteReidResult)(date, time));
electron_1.ipcMain.handle('renameReidGroup', (_, date, time, old_group_id, new_group_id) => (0, controller_1.renameReidGroup)(date, time, old_group_id, new_group_id));
electron_1.ipcMain.handle('terminateAI', (_) => (0, controller_1.terminateAI)());
electron_1.app.on('ready', () => {
    createWindow();
    electron_1.app.on('activate', function () {
        if (electron_1.BrowserWindow.getAllWindows().length === 0)
            createWindow();
    });
});
electron_1.app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        electron_1.app.quit();
    }
});
