"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.refreshDriveCache = refreshDriveCache;
exports.isRemovableDrive = isRemovableDrive;
exports.getLocalImagePath = getLocalImagePath;
exports.getUniqueFilename = getUniqueFilename;
const child_process_1 = require("child_process");
const util_1 = require("util");
const path_1 = __importDefault(require("path"));
const execAsync = (0, util_1.promisify)(child_process_1.exec);
// Cache for drive types (drive letter -> driveType number)
let driveTypeCache = new Map();
let lastCacheTime = 0;
const CACHE_TTL_MS = 30000; // 30 seconds
/**
 * Windows DriveType values from Win32_LogicalDisk:
 * 2 = Removable (USB, SD card)
 * 3 = Local Fixed Disk
 * 4 = Network Drive
 * 5 = CD-ROM
 * 6 = RAM Disk
 */
/**
 * Refresh the drive type cache by querying the system.
 * Call this when opening file dialogs or at app startup.
 */
async function refreshDriveCache() {
    if (process.platform === 'win32') {
        await refreshWindowsDriveCache();
    }
    // macOS doesn't need caching - we check mount points directly
    lastCacheTime = Date.now();
}
async function refreshWindowsDriveCache() {
    try {
        // Use PowerShell to query drive types
        const { stdout } = await execAsync('powershell -Command "Get-CimInstance Win32_LogicalDisk | Select-Object DeviceID, DriveType | ConvertTo-Json"', { timeout: 5000 });
        const drives = JSON.parse(stdout);
        driveTypeCache.clear();
        // Handle both single drive (object) and multiple drives (array)
        const driveList = Array.isArray(drives) ? drives : [drives];
        for (const drive of driveList) {
            if (drive.DeviceID && drive.DriveType !== undefined) {
                // Normalize to uppercase letter only (e.g., "C:" -> "C")
                const letter = drive.DeviceID.replace(':', '').toUpperCase();
                driveTypeCache.set(letter, drive.DriveType);
            }
        }
        console.log('[DriveType] Cache refreshed:', Object.fromEntries(driveTypeCache));
    }
    catch (error) {
        console.error('[DriveType] Failed to refresh Windows drive cache:', error);
    }
}
/**
 * Check if a file path is on a removable or network drive.
 * If true, files should be copied to local storage during import.
 */
async function isRemovableDrive(filePath) {
    // Ensure cache is fresh
    if (Date.now() - lastCacheTime > CACHE_TTL_MS) {
        await refreshDriveCache();
    }
    if (process.platform === 'win32') {
        return isRemovableWindows(filePath);
    }
    else if (process.platform === 'darwin') {
        return isRemovableMacOS(filePath);
    }
    // Linux/other: assume local for now
    return false;
}
function isRemovableWindows(filePath) {
    // Extract drive letter from path (e.g., "D:\folder\file.jpg" -> "D")
    const normalized = path_1.default.normalize(filePath);
    const match = normalized.match(/^([A-Za-z]):/);
    if (!match) {
        // UNC path or other format - treat as potentially network
        if (normalized.startsWith('\\\\')) {
            return true; // Network path
        }
        return false;
    }
    const driveLetter = match[1].toUpperCase();
    const driveType = driveTypeCache.get(driveLetter);
    if (driveType === undefined) {
        // Unknown drive - be conservative, don't copy
        console.warn(`[DriveType] Unknown drive type for ${driveLetter}:, assuming local`);
        return false;
    }
    // DriveType 2 = Removable, 4 = Network
    return driveType === 2 || driveType === 4;
}
function isRemovableMacOS(filePath) {
    const normalized = path_1.default.normalize(filePath);
    // Check if under /Volumes/ but not the main Macintosh HD
    if (normalized.startsWith('/Volumes/')) {
        // Extract volume name
        const parts = normalized.split('/');
        if (parts.length >= 3) {
            const volumeName = parts[2];
            // "Macintosh HD" is typically the main disk
            // Also check for common system volume names
            const systemVolumes = ['Macintosh HD', 'Macintosh HD - Data', 'System'];
            if (!systemVolumes.includes(volumeName)) {
                return true; // External volume
            }
        }
    }
    return false;
}
/**
 * Get the local storage path for a copied file.
 * Creates: data/images/{groupName}/{filename}
 */
function getLocalImagePath(originalPath, groupName, baseDir) {
    const filename = path_1.default.basename(originalPath);
    return path_1.default.join(baseDir, 'data', 'images', sanitizeFolderName(groupName), filename);
}
/**
 * Generate a unique filename if the target already exists.
 * image.jpg -> image_1.jpg -> image_2.jpg
 */
function getUniqueFilename(targetPath, existingFiles) {
    if (!existingFiles.has(targetPath)) {
        return targetPath;
    }
    const dir = path_1.default.dirname(targetPath);
    const ext = path_1.default.extname(targetPath);
    const base = path_1.default.basename(targetPath, ext);
    let counter = 1;
    let newPath = path_1.default.join(dir, `${base}_${counter}${ext}`);
    while (existingFiles.has(newPath)) {
        counter++;
        newPath = path_1.default.join(dir, `${base}_${counter}${ext}`);
    }
    return newPath;
}
function sanitizeFolderName(name) {
    // Remove or replace invalid characters for folder names
    return name.replace(/[<>:"/\\|?*]/g, '_').trim() || 'imported';
}
