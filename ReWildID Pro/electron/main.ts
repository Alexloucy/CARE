import { app, BrowserWindow, ipcMain, session, desktopCapturer, shell } from 'electron';
import path from 'path';
import {
    browseDetectImage,
    browseImage,
    browseReidImage,
    detect,
    deleteReidResult,
    downloadSelectedGalleryImages,
    downloadDetectImages,
    downloadReidImages,
    downloadSelectedDetectImages,
    getDetectImagePaths,
    getImagePaths,
    renameReidGroup,
    runReid,
    terminateAI,
    uploadImage,
    uploadPaths,
    viewDetectImage,
    viewImage,
    deleteGroup,
    deleteImage,
    updateGroupName,
    getImages,
    checkIsDirectory
} from './controller';

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
    app.quit();
}

let mainWindow: BrowserWindow | null = null;

function createWindow(): void {
    mainWindow = new BrowserWindow({
        width: 1280,
        height: 800,
        frame: false, // Make the window frameless
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js'),
            webSecurity: false, // Keeping consistent with neurolink/care-electron
            allowRunningInsecureContent: false,
        },
    });

    // Grant media permissions (covers webcam, microphone, and screen recording)
    mainWindow.webContents.session.setPermissionRequestHandler((webContents, permission, callback) => {
        if (permission === 'media') {
            callback(true);
        } else {
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
        desktopCapturer.getSources({ types: ['screen', 'window'] }).then((sources) => {
            const screenSource = sources.find(source => source.id.startsWith('screen:')) || sources[0];
            if (screenSource) {
                callback({ video: screenSource, audio: 'loopback' });
            } else {
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
    } else {
        mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
    }

    // Open external links in default browser
    mainWindow.webContents.setWindowOpenHandler((details) => {
        shell.openExternal(details.url);
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
const stream = (txt: string) => {
    if (mainWindow) {
        mainWindow.webContents.send('stream', txt);
    } else {
        console.log('null mainWindow, cannot send stream data');
    }
};

// IPC handlers for window controls
ipcMain.handle('window:minimize', () => {
    if (mainWindow) mainWindow.minimize();
});

ipcMain.handle('window:maximize', () => {
    if (mainWindow) {
        if (mainWindow.isMaximized()) {
            mainWindow.unmaximize();
        } else {
            mainWindow.maximize();
        }
    }
});

ipcMain.handle('window:close', () => {
    if (mainWindow) mainWindow.close();
});

ipcMain.handle('window:isMaximized', () => {
    return mainWindow ? mainWindow.isMaximized() : false;
});

// IPC handlers for backend logic
ipcMain.handle('browseImage', (_, date, folderPath) => browseImage(date, folderPath));
ipcMain.handle('viewImage', (_, originalPath) => viewImage(originalPath));
ipcMain.handle('getImagePaths', (_, currentFolder) => getImagePaths(currentFolder));
ipcMain.handle('getImages', (_, filter) => getImages(filter));
ipcMain.handle('downloadSelectedGalleryImages', (_, selectedPaths) => downloadSelectedGalleryImages(selectedPaths));
ipcMain.handle('uploadImage', (_, relativePath, originalPath) => uploadImage(relativePath, originalPath));
ipcMain.handle('uploadPaths', (_, filePaths, groupName) => uploadPaths(filePaths, groupName));
ipcMain.handle('deleteGroup', (_, id) => deleteGroup(id));
ipcMain.handle('deleteImage', (_, id) => deleteImage(id));
ipcMain.handle('updateGroupName', (_, id, name) => updateGroupName(id, name));
ipcMain.handle('checkIsDirectory', (_, filePath) => checkIsDirectory(filePath));
ipcMain.handle('detect', (_, selectedPaths) => detect(selectedPaths, stream));
ipcMain.handle('browseDetectImage', (_, date, folderPath, filterLabel, confLow, confHigh) =>
    browseDetectImage(date, folderPath, filterLabel, confLow, confHigh)
);
ipcMain.handle('viewDetectImage', (_, date, imagePath) => viewDetectImage(date, imagePath));
ipcMain.handle('getDetectImagePaths', (_, dirPath, filterLabel, confLow, confHigh) =>
    getDetectImagePaths(dirPath, filterLabel, confLow, confHigh)
);
ipcMain.handle('downloadDetectImages', (_, filterLabel) => downloadDetectImages(filterLabel));
ipcMain.handle('downloadSelectedDetectImages', (_, selectPaths) => downloadSelectedDetectImages(selectPaths));
ipcMain.handle('runReid', (_, selectedPaths) => runReid(selectedPaths, stream));
ipcMain.handle('browseReidImage', (_, date, time, group_id) => browseReidImage(date, time, group_id));
ipcMain.handle('downloadReidImages', (_, date, time) => downloadReidImages(date, time));
ipcMain.handle('deleteReidResult', (_, date, time) => deleteReidResult(date, time));
ipcMain.handle('renameReidGroup', (_, date, time, old_group_id, new_group_id) =>
    renameReidGroup(date, time, old_group_id, new_group_id)
);
ipcMain.handle('terminateAI', (_) => terminateAI());


app.on('ready', () => {
    createWindow();

    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
