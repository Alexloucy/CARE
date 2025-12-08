import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';

const DATA_DIR = path.join(process.cwd(), 'data');
const BACKUP_DIR = path.join(DATA_DIR, 'backups');
const DB_PATH = path.join(DATA_DIR, 'library.db');
const MAX_BACKUPS_PER_TABLE = 10;

// Ensure backup directory exists
function ensureBackupDir(): void {
    if (!fs.existsSync(BACKUP_DIR)) {
        fs.mkdirSync(BACKUP_DIR, { recursive: true });
    }
}

// Generate a unique backup filename
function generateBackupFilename(tableName: string): string {
    const now = new Date();
    const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const random = Math.random().toString(36).substring(2, 8);
    return `${tableName}_${timestamp}_${random}.json`;
}

export interface BackupInfo {
    filename: string;
    tableName: string;
    timestamp: Date;
    rowCount: number;
    path: string;
}

export interface BackupResult {
    success: boolean;
    backupPath?: string;
    rowCount?: number;
    error?: string;
}

/**
 * Backup rows from a table that match a WHERE clause.
 * If no whereClause provided, backs up entire table.
 */
export function backupTable(
    tableName: string,
    whereClause?: string,
    params?: any[]
): BackupResult {
    try {
        ensureBackupDir();

        const db = new Database(DB_PATH, { readonly: true });

        // Build query
        let query = `SELECT * FROM ${tableName}`;
        if (whereClause) {
            query += ` WHERE ${whereClause}`;
        }

        const rows = params ? db.prepare(query).all(...params) : db.prepare(query).all();
        db.close();

        if (rows.length === 0) {
            return { success: true, rowCount: 0 };
        }

        // Create backup file
        const filename = generateBackupFilename(tableName);
        const backupPath = path.join(BACKUP_DIR, filename);

        const backupData = {
            table: tableName,
            timestamp: new Date().toISOString(),
            whereClause: whereClause || null,
            rowCount: rows.length,
            rows: rows,
        };

        fs.writeFileSync(backupPath, JSON.stringify(backupData, null, 2));

        // Cleanup old backups
        cleanupOldBackups(tableName);

        console.log(`[Backup] Created backup: ${filename} (${rows.length} rows)`);

        return {
            success: true,
            backupPath,
            rowCount: rows.length,
        };
    } catch (error) {
        console.error('[Backup] Error:', error);
        return {
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error',
        };
    }
}

/**
 * List all backups for a table (or all tables if not specified)
 */
export function listBackups(tableName?: string): BackupInfo[] {
    ensureBackupDir();

    const files = fs.readdirSync(BACKUP_DIR).filter(f => f.endsWith('.json'));
    const backups: BackupInfo[] = [];

    for (const filename of files) {
        try {
            const filePath = path.join(BACKUP_DIR, filename);
            const content = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

            if (tableName && content.table !== tableName) {
                continue;
            }

            backups.push({
                filename,
                tableName: content.table,
                timestamp: new Date(content.timestamp),
                rowCount: content.rowCount || content.rows?.length || 0,
                path: filePath,
            });
        } catch (e) {
            console.error(`Failed to parse backup file ${filename}:`, e);
        }
    }

    // Sort by timestamp descending (newest first)
    backups.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    return backups;
}

/**
 * Restore data from a backup file.
 * This replaces/updates rows in the table based on their primary key.
 */
export function restoreBackup(backupPath: string): BackupResult {
    try {
        if (!fs.existsSync(backupPath)) {
            return { success: false, error: 'Backup file not found' };
        }

        const backupData = JSON.parse(fs.readFileSync(backupPath, 'utf-8'));
        const { table, rows } = backupData;

        if (!rows || rows.length === 0) {
            return { success: true, rowCount: 0 };
        }

        const db = new Database(DB_PATH);

        // Get column names from first row
        const columns = Object.keys(rows[0]);

        // Use INSERT OR REPLACE to restore rows
        const placeholders = columns.map(() => '?').join(', ');
        const columnNames = columns.join(', ');
        const stmt = db.prepare(
            `INSERT OR REPLACE INTO ${table} (${columnNames}) VALUES (${placeholders})`
        );

        const restoreTransaction = db.transaction((rowsToRestore: any[]) => {
            for (const row of rowsToRestore) {
                const values = columns.map(col => row[col]);
                stmt.run(...values);
            }
        });

        restoreTransaction(rows);
        db.close();

        console.log(`[Backup] Restored ${rows.length} rows to ${table} from ${backupPath}`);

        return {
            success: true,
            rowCount: rows.length,
        };
    } catch (error) {
        console.error('[Backup] Restore error:', error);
        return {
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error',
        };
    }
}

/**
 * Clean up old backups, keeping only the most recent N per table.
 */
function cleanupOldBackups(tableName: string): void {
    const backups = listBackups(tableName);

    if (backups.length <= MAX_BACKUPS_PER_TABLE) {
        return;
    }

    // Delete oldest backups beyond the limit
    const toDelete = backups.slice(MAX_BACKUPS_PER_TABLE);
    for (const backup of toDelete) {
        try {
            fs.unlinkSync(backup.path);
            console.log(`[Backup] Deleted old backup: ${backup.filename}`);
        } catch (e) {
            console.error(`Failed to delete backup ${backup.filename}:`, e);
        }
    }
}

/**
 * Delete a specific backup file
 */
export function deleteBackup(backupPath: string): BackupResult {
    try {
        if (fs.existsSync(backupPath)) {
            fs.unlinkSync(backupPath);
            return { success: true };
        }
        return { success: false, error: 'Backup file not found' };
    } catch (error) {
        return {
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error',
        };
    }
}
