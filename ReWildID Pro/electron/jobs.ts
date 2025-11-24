import { BrowserWindow, nativeImage } from 'electron';
import { randomUUID } from 'crypto';
import { DatabaseService } from './database';
import path from 'path';
import fs from 'fs-extra';
import os from 'os';
import { spawnPythonSubprocess, terminateSubprocess, setSubProcess } from './python';

function getAppDataDir() {
    if (process.platform === 'win32') {
        let appDataPath = process.env.APPDATA || process.env.LOCALAPPDATA
        if (appDataPath) {
            return path.join(appDataPath, 'ml4sg-care')
        }
    }
    return path.join(os.homedir(), '.ml4sg-care')
}

export interface Job {
    id: string;
    type: 'import' | 'thumbnail' | 'detect' | 'reid';
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    message: string;
    payload: any;
    createdAt: number;
    completedAt?: number;
    error?: string;
}

export class JobManager {
    private static instance: JobManager;
    private queue: Job[] = [];
    private activeJobs: Map<string, Job> = new Map();
    private completedJobs: Job[] = [];
    private mainWindow: BrowserWindow | null = null;
    private maxConcurrent = 2;
    private processing = false;
    private maxHistory = 50;

    private constructor() {}

    static getInstance(): JobManager {
        if (!JobManager.instance) {
            JobManager.instance = new JobManager();
        }
        return JobManager.instance;
    }

    setMainWindow(window: BrowserWindow) {
        this.mainWindow = window;
    }

    addJob(type: Job['type'], payload: any): string {
        const job: Job = {
            id: randomUUID(),
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

    getJobs(): Job[] {
        return [
            ...Array.from(this.activeJobs.values()), 
            ...this.queue,
            ...this.completedJobs
        ].sort((a, b) => b.createdAt - a.createdAt);
    }

    cancelJob(id: string) {
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

    private addToHistory(job: Job) {
        this.completedJobs.unshift(job);
        if (this.completedJobs.length > this.maxHistory) {
            this.completedJobs.pop();
        }
    }

    private emitUpdate() {
        if (this.mainWindow) {
            this.mainWindow.webContents.send('job-update', this.getJobs());
        }
    }

    private async processQueue() {
        if (this.processing) return;
        this.processing = true;

        try {
            while (this.activeJobs.size < this.maxConcurrent && this.queue.length > 0) {
                const job = this.queue.shift();
                if (!job) break;

                this.activeJobs.set(job.id, job);
                // Do not await here to allow concurrency
                this.runJob(job);
            }
        } finally {
            this.processing = false;
        }
    }

    private async runJob(job: Job) {
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
            
            if ((job.status as string) !== 'cancelled') {
                job.status = 'completed';
                job.progress = 100;
                // Only set default message if none was set by handler
                if (job.message === 'Starting...' || !job.message) {
                    job.message = 'Completed';
                }
            }
        } catch (error) {
            console.error(`Job ${job.id} failed:`, error);
            if ((job.status as string) !== 'cancelled') {
                job.status = 'failed';
                job.error = error instanceof Error ? error.message : String(error);
            }
        } finally {
            job.completedAt = Date.now();
            this.activeJobs.delete(job.id);
            this.addToHistory(job);
            this.emitUpdate();
            
            // Trigger next job
            this.processQueue();
        }
    }

    // --- Workers ---

    private async generateThumbnail(imageId: number, originalPath: string) {
        try {
            const thumbDir = path.join(process.cwd(), 'data', 'thumbnails');
            await fs.ensureDir(thumbDir);

            const thumbFilename = `${imageId}_thumb.jpg`;
            const thumbPath = path.join(thumbDir, thumbFilename);

            const image = nativeImage.createFromPath(originalPath);
            if (image.isEmpty()) {
                 return;
            }
            
            const resized = image.resize({ height: 300 });
            const buffer = resized.toJPEG(80);
            
            await fs.writeFile(thumbPath, buffer);
            DatabaseService.updateImagePreview(imageId, thumbPath);
        } catch (error) {
            console.error('Thumbnail generation failed:', error);
        }
    }

    private async countFiles(filePaths: string[]): Promise<number> {
        let count = 0;
        const processDir = async (dir: string) => {
            try {
                const files = await fs.readdir(dir);
                for (const file of files) {
                    const fullPath = path.join(dir, file);
                    const stat = await fs.stat(fullPath).catch(()=>null);
                    if (stat?.isDirectory()) await processDir(fullPath);
                    else if (stat?.isFile()) {
                         const ext = path.extname(file).toLowerCase();
                         if (ext === '.jpg' || ext === '.jpeg') count++;
                    }
                }
            } catch (e) { console.warn('Count error:', e); }
        };

        for (const p of filePaths) {
            const stat = await fs.stat(p).catch(()=>null);
            if (stat?.isDirectory()) await processDir(p);
            else if (stat?.isFile()) {
                const ext = path.extname(p).toLowerCase();
                if (ext === '.jpg' || ext === '.jpeg') count++;
            }
        }
        return count;
    }

    private async handleImportJob(job: Job) {
        const { filePaths, groupName } = job.payload;
        
        job.message = 'Scanning files...';
        this.emitUpdate();

        // Count total for progress
        const totalFiles = await this.countFiles(filePaths);
        
        // Create Group if needed (for flat file lists)
        let currentGroupId: number | null = null;
        
        // Pre-check: are we uploading a list of files directly?
        const filesOnly = [];
        for (const p of filePaths) {
            try {
                const stat = await fs.stat(p);
                if (stat.isFile()) filesOnly.push(p);
            } catch (e) {
                console.warn(`Failed to stat ${p}`, e);
            }
        }

        if (filesOnly.length > 0 && groupName) {
            currentGroupId = DatabaseService.createGroup(groupName);
        }

        // Recursive Process
        let processedCount = 0;
        
        const processFile = async (filePath: string, groupId: number) => {
            if ((job.status as string) === 'cancelled') return;
            
            const ext = path.extname(filePath).toLowerCase();
            if (ext === '.jpg' || ext === '.jpeg') {
                try {
                    // Add to DB
                    const imageId = DatabaseService.addImage(groupId, filePath);
                    
                    // Generate Thumbnail Synchronously (or await it) to keep it in one job
                    await this.generateThumbnail(imageId, filePath);
                    
                } catch (e) {
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

        const processDir = async (dirPath: string) => {
            if ((job.status as string) === 'cancelled') return;

            try {
                const stat = await fs.stat(dirPath);
                if (!stat.isDirectory()) return;

                const folderName = path.basename(dirPath);
                const groupId = DatabaseService.createGroup(folderName);

                const files = await fs.readdir(dirPath);
                for (const file of files) {
                    if ((job.status as string) === 'cancelled') return;
                    const fullPath = path.join(dirPath, file);
                    try {
                        const fileStat = await fs.stat(fullPath);
                        if (fileStat.isDirectory()) {
                            await processDir(fullPath);
                        } else if (fileStat.isFile()) {
                            await processFile(fullPath, groupId);
                        }
                    } catch (e) {
                         console.warn(`Error processing file ${fullPath}:`, e);
                    }
                }
            } catch (e) {
                console.warn(`Error processing dir ${dirPath}:`, e);
            }
        };

        // Start processing
        for (const p of filePaths) {
             if ((job.status as string) === 'cancelled') break;
             try {
                const stat = await fs.stat(p);
                if (stat.isDirectory()) {
                    await processDir(p);
                } else if (currentGroupId !== null && stat.isFile()) {
                    await processFile(p, currentGroupId);
                }
             } catch (e) {
                 console.warn(`Error accessing path ${p}:`, e);
             }
        }
        
        job.message = `Imported ${processedCount} images.`;
        job.progress = 100;
    }

    private async handleThumbnailJob(job: Job) {
        const { imageId, originalPath } = job.payload;
        await this.generateThumbnail(imageId, originalPath);
    }

    private async handleDetectJob(job: Job) {
        const { selectedPaths } = job.payload;
        // Use project root for data to keep it local
        const baseDataDir = process.cwd();
        // Create unique, deterministic output paths based on job ID
        const detectionJobDir = path.join(baseDataDir, 'data', 'detections', job.id);
        const imageOutputDir = path.join(detectionJobDir, 'images');
        const jsonOutputDir = path.join(detectionJobDir, 'json');
        
        const manifestPath = path.join(baseDataDir, 'data', 'temp', `detection_manifest_${job.id}.json`);

        try {
            terminateSubprocess();
            await fs.remove(manifestPath).catch(() => {});

            // Validate paths
            const absolutePaths: string[] = [];
            for (const imagePath of selectedPaths) {
                if (await fs.pathExists(imagePath)) {
                    absolutePaths.push(imagePath);
                }
            }

            if (absolutePaths.length === 0) {
                throw new Error('No valid images found to process.');
            }

            // Write Manifest
            await fs.ensureDir(path.dirname(manifestPath));
            await fs.writeJson(manifestPath, { files: absolutePaths }, { spaces: 2 });

            // Ensure output directories exist
            await fs.ensureDir(imageOutputDir);
            await fs.ensureDir(jsonOutputDir);

            // Spawn Python
            const args = [
                'detection',
                manifestPath,
                imageOutputDir,
                jsonOutputDir,
                path.join(baseDataDir, 'logs')
            ];

            const ps = spawnPythonSubprocess(args);
            setSubProcess(ps);

            if (!ps || !ps.stdout) {
                throw new Error('Failed to spawn Python process.');
            }

            job.message = 'Initializing AI models...';
            this.emitUpdate();

            // Wrap process in promise
            await new Promise<void>((resolve, reject) => {
                ps.stdout?.on('data', (data: Buffer) => {
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
                    } else if (txt.includes('Loading models')) {
                        job.message = 'Loading AI models...';
                        this.emitUpdate();
                    } else if (txt.includes('Running MegaDetector')) {
                         job.message = 'Running Object Detection...';
                         this.emitUpdate();
                    }
                });

                ps.on('close', (code) => {
                    setSubProcess(null);
                    if (code === 0) {
                        resolve();
                    } else {
                        reject(new Error(`Python process exited with code ${code}`));
                    }
                });

                ps.on('error', (err) => {
                    reject(err);
                });
            });

        } finally {
            // Cleanup
            await fs.remove(manifestPath).catch(() => {});
        }
    }
}
