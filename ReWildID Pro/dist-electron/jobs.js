"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.JobManager = void 0;
const electron_1 = require("electron");
const crypto_1 = require("crypto");
const database_1 = require("./database");
const path_1 = __importDefault(require("path"));
const fs_extra_1 = __importDefault(require("fs-extra"));
const os_1 = __importDefault(require("os"));
const python_1 = require("./python");
function getAppDataDir() {
    if (process.platform === 'win32') {
        let appDataPath = process.env.APPDATA || process.env.LOCALAPPDATA;
        if (appDataPath) {
            return path_1.default.join(appDataPath, 'ml4sg-care');
        }
    }
    return path_1.default.join(os_1.default.homedir(), '.ml4sg-care');
}
class JobManager {
    static instance;
    queue = [];
    activeJobs = new Map();
    completedJobs = [];
    mainWindow = null;
    maxConcurrent = 2;
    processing = false;
    maxHistory = 50;
    constructor() { }
    static getInstance() {
        if (!JobManager.instance) {
            JobManager.instance = new JobManager();
        }
        return JobManager.instance;
    }
    setMainWindow(window) {
        this.mainWindow = window;
    }
    addJob(type, payload) {
        const job = {
            id: (0, crypto_1.randomUUID)(),
            type,
            status: 'pending',
            progress: 0,
            message: 'Queued',
            payload,
            createdAt: Date.now()
        };
        this.queue.push(job);
        this.emitUpdate();
        this.processQueue();
        return job.id;
    }
    getJobs() {
        return [
            ...Array.from(this.activeJobs.values()),
            ...this.queue,
            ...this.completedJobs
        ].sort((a, b) => b.createdAt - a.createdAt);
    }
    cancelJob(id) {
        const queuedIndex = this.queue.findIndex(j => j.id === id);
        if (queuedIndex !== -1) {
            const job = this.queue[queuedIndex];
            job.status = 'cancelled';
            job.completedAt = Date.now();
            this.queue.splice(queuedIndex, 1);
            this.addToHistory(job);
            this.emitUpdate();
            return;
        }
        if (this.activeJobs.has(id)) {
            const job = this.activeJobs.get(id);
            if (job) {
                job.status = 'cancelled';
                // We cannot easily kill the async promise, but the job will check status
                this.emitUpdate();
            }
        }
    }
    addToHistory(job) {
        this.completedJobs.unshift(job);
        if (this.completedJobs.length > this.maxHistory) {
            this.completedJobs.pop();
        }
    }
    emitUpdate() {
        if (this.mainWindow) {
            this.mainWindow.webContents.send('job-update', this.getJobs());
        }
    }
    async processQueue() {
        if (this.processing)
            return;
        this.processing = true;
        try {
            while (this.activeJobs.size < this.maxConcurrent && this.queue.length > 0) {
                const job = this.queue.shift();
                if (!job)
                    break;
                this.activeJobs.set(job.id, job);
                // Do not await here to allow concurrency
                this.runJob(job);
            }
        }
        finally {
            this.processing = false;
        }
    }
    async runJob(job) {
        job.status = 'running';
        job.message = 'Starting...';
        this.emitUpdate();
        try {
            switch (job.type) {
                case 'import':
                    await this.handleImportJob(job);
                    break;
                case 'thumbnail':
                    await this.handleThumbnailJob(job);
                    break;
                case 'detect':
                    await this.handleDetectJob(job);
                    break;
                default:
                    throw new Error(`Unknown job type: ${job.type}`);
            }
            if (job.status !== 'cancelled') {
                job.status = 'completed';
                job.progress = 100;
                // Only set default message if none was set by handler
                if (job.message === 'Starting...' || !job.message) {
                    job.message = 'Completed';
                }
            }
        }
        catch (error) {
            console.error(`Job ${job.id} failed:`, error);
            if (job.status !== 'cancelled') {
                job.status = 'failed';
                job.error = error instanceof Error ? error.message : String(error);
            }
        }
        finally {
            job.completedAt = Date.now();
            this.activeJobs.delete(job.id);
            this.addToHistory(job);
            this.emitUpdate();
            // Trigger next job
            this.processQueue();
        }
    }
    // --- Workers ---
    async generateThumbnail(imageId, originalPath) {
        try {
            const thumbDir = path_1.default.join(process.cwd(), 'data', 'thumbnails');
            await fs_extra_1.default.ensureDir(thumbDir);
            const thumbFilename = `${imageId}_thumb.jpg`;
            const thumbPath = path_1.default.join(thumbDir, thumbFilename);
            const image = electron_1.nativeImage.createFromPath(originalPath);
            if (image.isEmpty()) {
                return;
            }
            const resized = image.resize({ height: 300 });
            const buffer = resized.toJPEG(80);
            await fs_extra_1.default.writeFile(thumbPath, buffer);
            database_1.DatabaseService.updateImagePreview(imageId, thumbPath);
        }
        catch (error) {
            console.error('Thumbnail generation failed:', error);
        }
    }
    async countFiles(filePaths) {
        let count = 0;
        const processDir = async (dir) => {
            try {
                const files = await fs_extra_1.default.readdir(dir);
                for (const file of files) {
                    const fullPath = path_1.default.join(dir, file);
                    const stat = await fs_extra_1.default.stat(fullPath).catch(() => null);
                    if (stat?.isDirectory())
                        await processDir(fullPath);
                    else if (stat?.isFile()) {
                        const ext = path_1.default.extname(file).toLowerCase();
                        if (ext === '.jpg' || ext === '.jpeg')
                            count++;
                    }
                }
            }
            catch (e) {
                console.warn('Count error:', e);
            }
        };
        for (const p of filePaths) {
            const stat = await fs_extra_1.default.stat(p).catch(() => null);
            if (stat?.isDirectory())
                await processDir(p);
            else if (stat?.isFile()) {
                const ext = path_1.default.extname(p).toLowerCase();
                if (ext === '.jpg' || ext === '.jpeg')
                    count++;
            }
        }
        return count;
    }
    async handleImportJob(job) {
        const { filePaths, groupName } = job.payload;
        job.message = 'Scanning files...';
        this.emitUpdate();
        // Count total for progress
        const totalFiles = await this.countFiles(filePaths);
        // Create Group if needed (for flat file lists)
        let currentGroupId = null;
        // Pre-check: are we uploading a list of files directly?
        const filesOnly = [];
        for (const p of filePaths) {
            try {
                const stat = await fs_extra_1.default.stat(p);
                if (stat.isFile())
                    filesOnly.push(p);
            }
            catch (e) {
                console.warn(`Failed to stat ${p}`, e);
            }
        }
        if (filesOnly.length > 0 && groupName) {
            currentGroupId = database_1.DatabaseService.createGroup(groupName);
        }
        // Recursive Process
        let processedCount = 0;
        const processFile = async (filePath, groupId) => {
            if (job.status === 'cancelled')
                return;
            const ext = path_1.default.extname(filePath).toLowerCase();
            if (ext === '.jpg' || ext === '.jpeg') {
                try {
                    // Add to DB
                    const imageId = database_1.DatabaseService.addImage(groupId, filePath);
                    // Generate Thumbnail Synchronously (or await it) to keep it in one job
                    await this.generateThumbnail(imageId, filePath);
                }
                catch (e) {
                    console.error(`Error adding image ${filePath}:`, e);
                }
                processedCount++;
            }
            // Update Progress
            if (totalFiles > 0) {
                job.progress = Math.floor((processedCount / totalFiles) * 100);
            }
            // Throttle updates to avoid spamming IPC
            if (processedCount % 5 === 0) {
                job.message = `Importing ${processedCount}/${totalFiles}...`;
                this.emitUpdate();
            }
        };
        const processDir = async (dirPath) => {
            if (job.status === 'cancelled')
                return;
            try {
                const stat = await fs_extra_1.default.stat(dirPath);
                if (!stat.isDirectory())
                    return;
                const folderName = path_1.default.basename(dirPath);
                const groupId = database_1.DatabaseService.createGroup(folderName);
                const files = await fs_extra_1.default.readdir(dirPath);
                for (const file of files) {
                    if (job.status === 'cancelled')
                        return;
                    const fullPath = path_1.default.join(dirPath, file);
                    try {
                        const fileStat = await fs_extra_1.default.stat(fullPath);
                        if (fileStat.isDirectory()) {
                            await processDir(fullPath);
                        }
                        else if (fileStat.isFile()) {
                            await processFile(fullPath, groupId);
                        }
                    }
                    catch (e) {
                        console.warn(`Error processing file ${fullPath}:`, e);
                    }
                }
            }
            catch (e) {
                console.warn(`Error processing dir ${dirPath}:`, e);
            }
        };
        // Start processing
        for (const p of filePaths) {
            if (job.status === 'cancelled')
                break;
            try {
                const stat = await fs_extra_1.default.stat(p);
                if (stat.isDirectory()) {
                    await processDir(p);
                }
                else if (currentGroupId !== null && stat.isFile()) {
                    await processFile(p, currentGroupId);
                }
            }
            catch (e) {
                console.warn(`Error accessing path ${p}:`, e);
            }
        }
        job.message = `Imported ${processedCount} images.`;
        job.progress = 100;
    }
    async handleThumbnailJob(job) {
        const { imageId, originalPath } = job.payload;
        await this.generateThumbnail(imageId, originalPath);
    }
    async handleDetectJob(job) {
        const { selectedPaths } = job.payload;
        const userIdFolder = '1';
        const userProfileDir = getAppDataDir();
        const manifestPath = path_1.default.join(userProfileDir, 'temp', `detection_manifest_${job.id}.json`);
        try {
            (0, python_1.terminateSubprocess)();
            await fs_extra_1.default.remove(manifestPath).catch(() => { });
            // Validate paths
            const absolutePaths = [];
            for (const imagePath of selectedPaths) {
                if (await fs_extra_1.default.pathExists(imagePath)) {
                    absolutePaths.push(imagePath);
                }
            }
            if (absolutePaths.length === 0) {
                throw new Error('No valid images found to process.');
            }
            // Write Manifest
            await fs_extra_1.default.ensureDir(path_1.default.dirname(manifestPath));
            await fs_extra_1.default.writeJson(manifestPath, { files: absolutePaths }, { spaces: 2 });
            // Spawn Python
            const args = [
                'detection',
                manifestPath,
                path_1.default.join(userProfileDir, 'data/image_marked', userIdFolder),
                path_1.default.join(userProfileDir, 'data/image_cropped_json', userIdFolder),
                path_1.default.join(userProfileDir, 'logs')
            ];
            const ps = (0, python_1.spawnPythonSubprocess)(args);
            (0, python_1.setSubProcess)(ps);
            if (!ps || !ps.stdout) {
                throw new Error('Failed to spawn Python process.');
            }
            job.message = 'Initializing AI models...';
            this.emitUpdate();
            // Wrap process in promise
            await new Promise((resolve, reject) => {
                ps.stdout?.on('data', (data) => {
                    const txt = data.toString();
                    console.log(`[Job ${job.id}] ${txt.trim()}`);
                    // Parse progress
                    // Example: [1] PROCESS: 8/61
                    const processMatch = txt.match(/PROCESS:\s*(\d+)\/(\d+)/);
                    if (processMatch) {
                        const current = parseInt(processMatch[1]);
                        const total = parseInt(processMatch[2]);
                        if (total > 0) {
                            job.progress = Math.floor((current / total) * 100);
                            job.message = `Processing detections: ${current}/${total}`;
                            // Throttle updates slightly? JobManager.emitUpdate handles some UI
                            this.emitUpdate();
                        }
                    }
                    else if (txt.includes('Loading models')) {
                        job.message = 'Loading AI models...';
                        this.emitUpdate();
                    }
                    else if (txt.includes('Running MegaDetector')) {
                        job.message = 'Running Object Detection...';
                        this.emitUpdate();
                    }
                });
                ps.on('close', (code) => {
                    (0, python_1.setSubProcess)(null);
                    if (code === 0) {
                        resolve();
                    }
                    else {
                        reject(new Error(`Python process exited with code ${code}`));
                    }
                });
                ps.on('error', (err) => {
                    reject(err);
                });
            });
        }
        finally {
            // Cleanup
            await fs_extra_1.default.remove(manifestPath).catch(() => { });
        }
    }
}
exports.JobManager = JobManager;
