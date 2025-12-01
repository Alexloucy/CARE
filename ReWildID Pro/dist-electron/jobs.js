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
                case 'reid':
                    await this.handleReidJob(job);
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
        const { filePaths, groupName, afterAction, species } = job.payload;
        // Track imported image IDs for chained actions
        const importedImageIds = [];
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
                    importedImageIds.push(imageId);
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
        // Handle chained actions
        if (afterAction && importedImageIds.length > 0 && job.status !== 'cancelled') {
            if (afterAction === 'classify') {
                // Get paths for the imported images
                const images = database_1.DatabaseService.getImagesByIds(importedImageIds);
                const selectedPaths = images.map(img => img.original_path);
                job.message = `Imported ${processedCount} images. Starting classification...`;
                this.emitUpdate();
                // Queue a detect job
                this.addJob('detect', { selectedPaths });
            }
            else if (afterAction === 'reid' && species) {
                job.message = `Imported ${processedCount} images. Starting ReID...`;
                this.emitUpdate();
                // Queue a reid job with the imported image IDs
                this.addJob('reid', { imageIds: importedImageIds, species });
            }
        }
    }
    async handleThumbnailJob(job) {
        const { imageId, originalPath } = job.payload;
        await this.generateThumbnail(imageId, originalPath);
    }
    /**
     * Run detection inline (used by ReID job when images need detection first)
     * @param imageIdsToDetect - The actual image IDs from the database
     */
    async runDetectionInline(job, imageIdsToDetect) {
        const baseDataDir = process.cwd();
        const detectionJobDir = path_1.default.join(baseDataDir, 'data', 'detections', `reid_${job.id}`);
        const imageOutputDir = path_1.default.join(detectionJobDir, 'images');
        const jsonOutputDir = path_1.default.join(detectionJobDir, 'json');
        const manifestPath = path_1.default.join(baseDataDir, 'data', 'temp', `detection_manifest_reid_${job.id}.json`);
        try {
            (0, python_1.terminateSubprocess)();
            await fs_extra_1.default.remove(manifestPath).catch(() => { });
            // Get images from database - this gives us the correct ID -> path mapping
            const images = database_1.DatabaseService.getImagesByIds(imageIdsToDetect);
            // Build path -> id mapping for later
            const pathToIdMap = new Map();
            const absolutePaths = [];
            for (const img of images) {
                if (await fs_extra_1.default.pathExists(img.original_path)) {
                    absolutePaths.push(img.original_path);
                    // Map by filename since that's what we'll have in JSON output
                    const filename = path_1.default.parse(img.original_path).name;
                    pathToIdMap.set(filename, img.id);
                }
            }
            if (absolutePaths.length === 0) {
                throw new Error('No valid images found for detection.');
            }
            // Write Manifest
            await fs_extra_1.default.ensureDir(path_1.default.dirname(manifestPath));
            await fs_extra_1.default.writeJson(manifestPath, { files: absolutePaths }, { spaces: 2 });
            // Ensure output directories exist
            await fs_extra_1.default.ensureDir(imageOutputDir);
            await fs_extra_1.default.ensureDir(jsonOutputDir);
            // Spawn Python
            const args = [
                'detection',
                manifestPath,
                imageOutputDir,
                jsonOutputDir,
                path_1.default.join(baseDataDir, 'logs')
            ];
            const ps = (0, python_1.spawnPythonSubprocess)(args);
            (0, python_1.setSubProcess)(ps);
            if (!ps || !ps.stdout) {
                throw new Error('Failed to spawn Python process for detection.');
            }
            // Wrap process in promise
            await new Promise((resolve, reject) => {
                ps.stdout?.on('data', (data) => {
                    const txt = data.toString();
                    console.log(`[ReID Detection ${job.id}] ${txt.trim()}`);
                    // Parse progress
                    const processMatch = txt.match(/PROCESS:\s*(\d+)\/(\d+)/);
                    if (processMatch) {
                        const current = parseInt(processMatch[1]);
                        const total = parseInt(processMatch[2]);
                        if (total > 0) {
                            // Use 0-50% for detection, 50-100% for ReID
                            job.progress = Math.floor((current / total) * 50);
                            job.message = `Classification: ${current}/${total}`;
                            this.emitUpdate();
                        }
                    }
                    else if (txt.includes('Loading models')) {
                        job.message = 'Loading classification models...';
                        this.emitUpdate();
                    }
                });
                ps.on('close', (code) => {
                    (0, python_1.setSubProcess)(null);
                    if (code === 0) {
                        resolve();
                    }
                    else {
                        reject(new Error(`Detection process exited with code ${code}`));
                    }
                });
                ps.on('error', (err) => {
                    reject(err);
                });
            });
            // Import detection results to Database using the ID mapping
            const now = new Date();
            const dateStr = now.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
            const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const batchName = `ReID Pre-Detection ${dateStr} ${timeStr}`;
            const batchId = database_1.DatabaseService.createDetectionBatch(batchName);
            // Read all JSON files in the output directory
            const jsonFiles = await fs_extra_1.default.readdir(jsonOutputDir);
            for (const jsonFile of jsonFiles) {
                if (!jsonFile.endsWith('.json'))
                    continue;
                const jsonPath = path_1.default.join(jsonOutputDir, jsonFile);
                const baseName = path_1.default.parse(jsonFile).name;
                const imageId = pathToIdMap.get(baseName);
                if (!imageId) {
                    console.warn(`[ReID Detection] No image ID found for ${baseName}`);
                    continue;
                }
                try {
                    const result = await fs_extra_1.default.readJson(jsonPath);
                    if (result.boxes && Array.isArray(result.boxes)) {
                        for (const box of result.boxes) {
                            if (box.bbox && box.bbox.length === 4) {
                                database_1.DatabaseService.addDetection(batchId, imageId, box.label, box.pred_conf || 0, box.detection_conf || 0, box.bbox, box.source || 'unknown');
                            }
                        }
                    }
                }
                catch (e) {
                    console.error(`Failed to parse detection result for ${jsonFile}`, e);
                }
            }
            console.log(`[ReID Detection] Saved detections for ${jsonFiles.length} images to batch ${batchId}`);
        }
        finally {
            // Cleanup
            await fs_extra_1.default.remove(manifestPath).catch(() => { });
        }
    }
    async handleReidJob(job) {
        const { imageIds, species } = job.payload;
        const baseDataDir = process.cwd();
        const tempDir = path_1.default.join(baseDataDir, 'temp', 'reid_v2');
        try {
            await fs_extra_1.default.ensureDir(tempDir);
            job.message = 'Checking images for detections...';
            this.emitUpdate();
            // Step 1: Get images without detections
            const imagesWithoutDetections = database_1.DatabaseService.getImagesWithoutDetections(imageIds);
            // Step 2: If images need detection, run it first
            if (imagesWithoutDetections.length > 0) {
                job.message = `Running classification on ${imagesWithoutDetections.length} images first...`;
                this.emitUpdate();
                // Run detection inline with image IDs
                await this.runDetectionInline(job, imagesWithoutDetections);
                job.message = 'Classification complete. Starting ReID...';
                this.emitUpdate();
            }
            // Step 3: Get LATEST detections for selected images (only from most recent batch per image)
            const allDetections = database_1.DatabaseService.getLatestDetectionsForImages(imageIds);
            console.log(`[ReID Debug] imageIds: ${JSON.stringify(imageIds)}`);
            console.log(`[ReID Debug] allDetections count (latest batch only): ${allDetections.length}`);
            console.log(`[ReID Debug] allDetections labels: ${JSON.stringify(allDetections.map((d) => d.label))}`);
            // Step 4: Filter by species
            const speciesLower = species.toLowerCase();
            console.log(`[ReID Debug] Looking for species: "${speciesLower}"`);
            const matchingDetections = allDetections.filter((det) => det.label?.toLowerCase() === speciesLower);
            console.log(`[ReID Debug] matchingDetections count: ${matchingDetections.length}`);
            if (matchingDetections.length === 0) {
                throw new Error(`No ${species} detections found in the selected images. Found ${allDetections.length} detections with labels: ${[...new Set(allDetections.map((d) => d.label))].join(', ')}`);
            }
            job.message = `Found ${matchingDetections.length} ${species} detections. Starting ReID...`;
            this.emitUpdate();
            // Step 4: Generate input JSON for Python
            const inputJsonPath = path_1.default.join(tempDir, `reid_input_${job.id}.json`);
            const outputJsonPath = path_1.default.join(tempDir, `reid_output_${job.id}.json`);
            const inputData = {
                detections: matchingDetections.map((det) => ({
                    detection_id: det.id,
                    image_path: det.image_path,
                    bbox: [det.x1, det.y1, det.x2, det.y2]
                })),
                output_path: outputJsonPath
            };
            await fs_extra_1.default.writeJson(inputJsonPath, inputData, { spaces: 2 });
            // Step 5: Run Python reid_v2
            const args = ['reid_v2', inputJsonPath];
            const ps = (0, python_1.spawnPythonSubprocess)(args);
            if (!ps) {
                throw new Error('Failed to start Python process');
            }
            (0, python_1.setSubProcess)(ps);
            job.message = 'Loading AI models...';
            this.emitUpdate();
            // Wait for completion
            await new Promise((resolve, reject) => {
                ps.stdout?.on('data', (data) => {
                    const txt = data.toString();
                    console.log(`[ReID Job ${job.id}] ${txt.trim()}`);
                    // Parse progress
                    const processMatch = txt.match(/PROCESS:\s*(\d+)\/(\d+)/);
                    if (processMatch) {
                        const current = parseInt(processMatch[1]);
                        const total = parseInt(processMatch[2]);
                        if (total > 0) {
                            // Use 50-100% range for ReID (0-50% was detection)
                            job.progress = 50 + Math.floor((current / total) * 50);
                            job.message = `ReID: ${current}/${total}`;
                            this.emitUpdate();
                        }
                    }
                    else if (txt.includes('Loading model')) {
                        job.message = 'Loading ReID models...';
                        job.progress = 50;
                        this.emitUpdate();
                    }
                    else if (txt.includes('STATUS: PROCESSING')) {
                        job.message = 'Computing embeddings...';
                        this.emitUpdate();
                    }
                });
                ps.on('close', (code) => {
                    (0, python_1.setSubProcess)(null);
                    if (code === 0) {
                        resolve();
                    }
                    else {
                        reject(new Error(`ReID process exited with code ${code}`));
                    }
                });
                ps.on('error', (err) => {
                    reject(err);
                });
            });
            // Step 6: Parse output and store in database
            if (!await fs_extra_1.default.pathExists(outputJsonPath)) {
                throw new Error('ReID output file not found.');
            }
            const outputData = await fs_extra_1.default.readJson(outputJsonPath);
            // Create ReID run
            const now = new Date();
            const dateStr = now.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
            const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const runName = `ReID ${species} ${dateStr} ${timeStr}`;
            const reidRunId = database_1.DatabaseService.createReidRun(runName, species);
            // Create individuals and members
            for (const individual of outputData.individuals) {
                const individualId = database_1.DatabaseService.createReidIndividual(reidRunId, individual.name);
                for (const detectionId of individual.detection_ids) {
                    database_1.DatabaseService.addReidMember(individualId, detectionId);
                }
            }
            job.message = `Identified ${outputData.individuals.length} individuals`;
            // Cleanup temp files
            await fs_extra_1.default.remove(inputJsonPath).catch(() => { });
            await fs_extra_1.default.remove(outputJsonPath).catch(() => { });
        }
        catch (error) {
            throw error;
        }
    }
    async handleDetectJob(job) {
        const { selectedPaths } = job.payload;
        // Use project root for data to keep it local
        const baseDataDir = process.cwd();
        // Create unique, deterministic output paths based on job ID
        const detectionJobDir = path_1.default.join(baseDataDir, 'data', 'detections', job.id);
        const imageOutputDir = path_1.default.join(detectionJobDir, 'images');
        const jsonOutputDir = path_1.default.join(detectionJobDir, 'json');
        const manifestPath = path_1.default.join(baseDataDir, 'data', 'temp', `detection_manifest_${job.id}.json`);
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
            // Ensure output directories exist
            await fs_extra_1.default.ensureDir(imageOutputDir);
            await fs_extra_1.default.ensureDir(jsonOutputDir);
            // Spawn Python
            const args = [
                'detection',
                manifestPath,
                imageOutputDir,
                jsonOutputDir,
                path_1.default.join(baseDataDir, 'logs')
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
            // Post-process: Import results to Database
            // job.message = 'Saving results to database...'; // Kept internal, user sees progress bar
            this.emitUpdate();
            const now = new Date();
            const dateStr = now.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
            const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const batchName = `Detection ${dateStr} ${timeStr}`;
            const batchId = database_1.DatabaseService.createDetectionBatch(batchName);
            for (const originalPath of absolutePaths) {
                const filename = path_1.default.basename(originalPath);
                const jsonFilename = path_1.default.parse(filename).name + '.json';
                const jsonPath = path_1.default.join(jsonOutputDir, jsonFilename);
                if (await fs_extra_1.default.pathExists(jsonPath)) {
                    try {
                        const result = await fs_extra_1.default.readJson(jsonPath);
                        const image = database_1.DatabaseService.getImageByPath(originalPath);
                        if (image && result.boxes && Array.isArray(result.boxes)) {
                            for (const box of result.boxes) {
                                // Skip if label is null (no detection) unless we want to track "empty"
                                // Based on schema, label is nullable, so we can store it.
                                // But "no detection" usually means empty box list in some formats, or a specific "empty" entry.
                                // detection_utils outputs: { label: null, confidence: 0, bbox: [] } for no detection.
                                if (box.bbox && box.bbox.length === 4) {
                                    database_1.DatabaseService.addDetection(batchId, image.id, box.label, box.pred_conf || 0, box.detection_conf || 0, box.bbox, // [x1, y1, x2, y2]
                                    box.source || 'unknown');
                                }
                            }
                        }
                    }
                    catch (e) {
                        console.error(`Failed to parse result for ${originalPath}`, e);
                    }
                }
            }
        }
        finally {
            // Cleanup
            await fs_extra_1.default.remove(manifestPath).catch(() => { });
        }
    }
}
exports.JobManager = JobManager;
