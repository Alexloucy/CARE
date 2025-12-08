import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';

const execAsync = promisify(exec);

// Cache for drive types (drive letter -> driveType number)
let driveTypeCache: Map<string, number> = new Map();
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
export async function refreshDriveCache(): Promise<void> {
    if (process.platform === 'win32') {
        await refreshWindowsDriveCache();
    }
    // macOS doesn't need caching - we check mount points directly
    lastCacheTime = Date.now();
}

async function refreshWindowsDriveCache(): Promise<void> {
    try {
        // Use PowerShell to query drive types
        const { stdout } = await execAsync(
            'powershell -Command "Get-CimInstance Win32_LogicalDisk | Select-Object DeviceID, DriveType | ConvertTo-Json"',
            { timeout: 5000 }
        );

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
    } catch (error) {
        console.error('[DriveType] Failed to refresh Windows drive cache:', error);
    }
}

/**
 * Check if a file path is on a removable or network drive.
 * If true, files should be copied to local storage during import.
 */
export async function isRemovableDrive(filePath: string): Promise<boolean> {
    // Ensure cache is fresh
    if (Date.now() - lastCacheTime > CACHE_TTL_MS) {
        await refreshDriveCache();
    }

    if (process.platform === 'win32') {
        return isRemovableWindows(filePath);
    } else if (process.platform === 'darwin') {
        return isRemovableMacOS(filePath);
    }

    // Linux/other: assume local for now
    return false;
}

function isRemovableWindows(filePath: string): boolean {
    // Extract drive letter from path (e.g., "D:\folder\file.jpg" -> "D")
    const normalized = path.normalize(filePath);
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

function isRemovableMacOS(filePath: string): boolean {
    const normalized = path.normalize(filePath);

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
export function getLocalImagePath(originalPath: string, groupName: string, baseDir: string): string {
    const filename = path.basename(originalPath);
    return path.join(baseDir, 'data', 'images', sanitizeFolderName(groupName), filename);
}

/**
 * Generate a unique filename if the target already exists.
 * image.jpg -> image_1.jpg -> image_2.jpg
 */
export function getUniqueFilename(targetPath: string, existingFiles: Set<string>): string {
    if (!existingFiles.has(targetPath)) {
        return targetPath;
    }

    const dir = path.dirname(targetPath);
    const ext = path.extname(targetPath);
    const base = path.basename(targetPath, ext);

    let counter = 1;
    let newPath = path.join(dir, `${base}_${counter}${ext}`);

    while (existingFiles.has(newPath)) {
        counter++;
        newPath = path.join(dir, `${base}_${counter}${ext}`);
    }

    return newPath;
}

function sanitizeFolderName(name: string): string {
    // Remove or replace invalid characters for folder names
    return name.replace(/[<>:"/\\|?*]/g, '_').trim() || 'imported';
}
