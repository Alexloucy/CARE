"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
electron_1.contextBridge.exposeInMainWorld('windowControls', {
    minimize: () => electron_1.ipcRenderer.invoke('window:minimize'),
    maximize: () => electron_1.ipcRenderer.invoke('window:maximize'),
    close: () => electron_1.ipcRenderer.invoke('window:close'),
    isMaximized: () => electron_1.ipcRenderer.invoke('window:isMaximized'),
    onStateChange: (callback) => {
        const handler = (_event, { isMaximized }) => callback(isMaximized);
        electron_1.ipcRenderer.on('window-state-changed', handler);
        return () => electron_1.ipcRenderer.removeListener('window-state-changed', handler);
    },
    removeStateChangeListener: () => {
        electron_1.ipcRenderer.removeAllListeners('window-state-changed');
    }
});
electron_1.contextBridge.exposeInMainWorld('api', {
    browseImage: (date, folderPath) => electron_1.ipcRenderer.invoke('browseImage', date, folderPath),
    viewImage: (originalPath) => electron_1.ipcRenderer.invoke('viewImage', originalPath),
    getImagePaths: (currentFolder) => electron_1.ipcRenderer.invoke('getImagePaths', currentFolder),
    getImages: (filter) => electron_1.ipcRenderer.invoke('getImages', filter),
    uploadPaths: (filePaths, groupName) => electron_1.ipcRenderer.invoke('uploadPaths', filePaths, groupName),
    deleteGroup: (id) => electron_1.ipcRenderer.invoke('deleteGroup', id),
    deleteImage: (id) => electron_1.ipcRenderer.invoke('deleteImage', id),
    updateGroupName: (id, name) => electron_1.ipcRenderer.invoke('updateGroupName', id, name),
    checkIsDirectory: (filePath) => electron_1.ipcRenderer.invoke('checkIsDirectory', filePath),
    openFileDialog: () => electron_1.ipcRenderer.invoke('openFileDialog'),
    saveImages: (sourcePaths) => electron_1.ipcRenderer.invoke('saveImages', sourcePaths),
    detect: (selectedPaths, onStream) => {
        // Remove existing listeners to avoid duplicates
        electron_1.ipcRenderer.removeAllListeners('stream');
        electron_1.ipcRenderer.on('stream', (_, txt) => onStream(txt));
        return electron_1.ipcRenderer.invoke('detect', selectedPaths);
    },
    browseDetectImage: (date, folderPath, filterLabel, confLow, confHigh) => electron_1.ipcRenderer.invoke('browseDetectImage', date, folderPath, filterLabel, confLow, confHigh),
    viewDetectImage: (date, imagePath) => electron_1.ipcRenderer.invoke('viewDetectImage', date, imagePath),
    getDetectImagePaths: (dirPath, filterLabel, confLow, confHigh) => electron_1.ipcRenderer.invoke('getDetectImagePaths', dirPath, filterLabel, confLow, confHigh),
    downloadDetectImages: (filterLabel) => electron_1.ipcRenderer.invoke('downloadDetectImages', filterLabel),
    downloadSelectedDetectImages: (selectPaths) => electron_1.ipcRenderer.invoke('downloadSelectedDetectImages', selectPaths),
    runReid: (selectedPaths, onStream) => {
        electron_1.ipcRenderer.removeAllListeners('stream');
        electron_1.ipcRenderer.on('stream', (_, txt) => onStream(txt));
        return electron_1.ipcRenderer.invoke('runReid', selectedPaths);
    },
    browseReidImage: (date, time, group_id) => electron_1.ipcRenderer.invoke('browseReidImage', date, time, group_id),
    downloadReidImages: (date, time) => electron_1.ipcRenderer.invoke('downloadReidImages', date, time),
    deleteReidResult: (date, time) => electron_1.ipcRenderer.invoke('deleteReidResult', date, time),
    renameReidGroup: (date, time, old_group_id, new_group_id) => electron_1.ipcRenderer.invoke('renameReidGroup', date, time, old_group_id, new_group_id),
    terminateAI: () => electron_1.ipcRenderer.invoke('terminateAI'),
    getPathForFile: (file) => electron_1.webUtils.getPathForFile(file),
    // Jobs
    getJobs: () => electron_1.ipcRenderer.invoke('getJobs'),
    cancelJob: (id) => electron_1.ipcRenderer.invoke('cancelJob', id),
    onJobUpdate: (callback) => {
        const handler = (_event, jobs) => callback(jobs);
        electron_1.ipcRenderer.on('job-update', handler);
        return () => electron_1.ipcRenderer.removeListener('job-update', handler);
    }
});
