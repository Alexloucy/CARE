import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';
import { app } from 'electron';

const isDev = process.env.NODE_ENV === 'development';

// Determine database path
// In production, we might want to store it in appData, but for now adhering to process.cwd()/data as per previous logic
const DATA_DIR = path.join(process.cwd(), 'data');
const DB_PATH = path.join(DATA_DIR, 'library.db');

if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
}

const db = new Database(DB_PATH, { verbose: isDev ? console.log : undefined });
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON'); // Important for ON DELETE CASCADE

// Initialize Schema
const initSchema = () => {
    const createGroupsTable = `
        CREATE TABLE IF NOT EXISTS groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
    `;

    const createImagesTable = `
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INTEGER NOT NULL,
            original_path TEXT NOT NULL,
            preview_path TEXT,
            date_added INTEGER NOT NULL,
            FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE
        );
    `;

    const createDetectionBatchesTable = `
        CREATE TABLE IF NOT EXISTS detection_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
    `;

    const createDetectionsTable = `
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            image_id INTEGER NOT NULL,
            label TEXT,
            confidence REAL,
            detection_confidence REAL,
            x1 REAL,
            y1 REAL,
            x2 REAL,
            y2 REAL,
            source TEXT,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(batch_id) REFERENCES detection_batches(id) ON DELETE CASCADE,
            FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
        );
    `;

    db.exec(createGroupsTable);
    db.exec(createImagesTable);
    db.exec(createDetectionBatchesTable);
    db.exec(createDetectionsTable);
};

initSchema();

export interface Group {
    id: number;
    name: string;
    created_at: number;
    updated_at: number;
}

export interface Image {
    id: number;
    group_id: number;
    original_path: string;
    preview_path?: string;
    date_added: number;
}

export interface DetectionBatch {
    id: number;
    name: string;
    created_at: number;
    updated_at: number;
}

export interface Detection {
    id: number;
    batch_id: number;
    image_id: number;
    label: string;
    confidence: number;
    detection_confidence: number;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    source: string;
    created_at: number;
}

export const DatabaseService = {
    // --- Groups ---

    createGroup: (name: string, createdAt?: number): number => {
        const stmt = db.prepare('INSERT INTO groups (name, created_at, updated_at) VALUES (?, ?, ?)');
        const now = Date.now();
        const info = stmt.run(name, createdAt || now, now);
        return info.lastInsertRowid as number;
    },

    getGroup: (id: number): Group | undefined => {
        const stmt = db.prepare('SELECT * FROM groups WHERE id = ?');
        return stmt.get(id) as Group | undefined;
    },

    updateGroupName: (id: number, name: string): void => {
        const stmt = db.prepare('UPDATE groups SET name = ?, updated_at = ? WHERE id = ?');
        stmt.run(name, Date.now(), id);
    },

    deleteGroup: (id: number): void => {
        const stmt = db.prepare('DELETE FROM groups WHERE id = ?');
        stmt.run(id);
    },

    getAllGroups: (): Group[] => {
        const stmt = db.prepare('SELECT * FROM groups ORDER BY created_at DESC');
        return stmt.all() as Group[];
    },

    // --- Images ---

    addImage: (groupId: number, originalPath: string, previewPath?: string): number => {
        const stmt = db.prepare('INSERT INTO images (group_id, original_path, preview_path, date_added) VALUES (?, ?, ?, ?)');
        const info = stmt.run(groupId, originalPath, previewPath || null, Date.now());
        return info.lastInsertRowid as number;
    },

    updateImagePreview: (id: number, previewPath: string): void => {
        const stmt = db.prepare('UPDATE images SET preview_path = ? WHERE id = ?');
        stmt.run(previewPath, id);
    },

    deleteImage: (id: number): void => {
        const stmt = db.prepare('DELETE FROM images WHERE id = ?');
        stmt.run(id);
    },

    getImageByPath: (originalPath: string): Image | undefined => {
        const stmt = db.prepare('SELECT * FROM images WHERE original_path = ?');
        return stmt.get(originalPath) as Image | undefined;
    },

    getImages: (filter?: { date?: string, groupIds?: number[], searchQuery?: string }): (Image & { group_name: string, group_created_at: number })[] => {
        let query = `
            SELECT images.*, groups.name as group_name, groups.created_at as group_created_at
            FROM images
            JOIN groups ON images.group_id = groups.id
            WHERE 1=1
        `;
        const params: any[] = [];

        if (filter?.date) {
            // date string YYYYMMDD
            query += ` AND strftime('%Y%m%d', datetime(groups.created_at / 1000, 'unixepoch', 'localtime')) = ?`;
            params.push(filter.date);
        }

        if (filter?.groupIds && filter.groupIds.length > 0) {
            const placeholders = filter.groupIds.map(() => '?').join(',');
            query += ` AND groups.id IN (${placeholders})`;
            params.push(...filter.groupIds);
        }

        if (filter?.searchQuery) {
            query += ` AND (
                images.original_path LIKE ? OR 
                groups.name LIKE ?
            )`; 
            const likeQuery = `%${filter.searchQuery}%`;
            params.push(likeQuery, likeQuery);
        }

        query += ` ORDER BY groups.created_at DESC, images.date_added DESC`;

        const stmt = db.prepare(query);
        return stmt.all(...params) as (Image & { group_name: string, group_created_at: number })[];
    },

    // --- Detection Batches ---

    createDetectionBatch: (name: string): number => {
        const stmt = db.prepare('INSERT INTO detection_batches (name, created_at, updated_at) VALUES (?, ?, ?)');
        const now = Date.now();
        const info = stmt.run(name, now, now);
        return info.lastInsertRowid as number;
    },

    getDetectionBatches: (): DetectionBatch[] => {
        const stmt = db.prepare('SELECT * FROM detection_batches ORDER BY created_at DESC');
        return stmt.all() as DetectionBatch[];
    },

    updateDetectionBatchName: (id: number, name: string): void => {
        const stmt = db.prepare('UPDATE detection_batches SET name = ?, updated_at = ? WHERE id = ?');
        stmt.run(name, Date.now(), id);
    },

    deleteDetectionBatch: (id: number): void => {
        const stmt = db.prepare('DELETE FROM detection_batches WHERE id = ?');
        stmt.run(id);
    },

    // --- Detections ---

    addDetection: (
        batchId: number,
        imageId: number,
        label: string | null,
        confidence: number,
        detectionConfidence: number,
        bbox: [number, number, number, number],
        source: string
    ): number => {
        const stmt = db.prepare(`
            INSERT INTO detections (
                batch_id, image_id, label, confidence, detection_confidence, 
                x1, y1, x2, y2, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `);
        const now = Date.now();
        const info = stmt.run(
            batchId,
            imageId,
            label,
            confidence,
            detectionConfidence,
            bbox[0], bbox[1], bbox[2], bbox[3],
            source,
            now
        );
        return info.lastInsertRowid as number;
    },

    getDetectionsForBatch: (batchId: number): (Detection & Image)[] => {
        const stmt = db.prepare(`
            SELECT detections.*, images.*
            FROM detections
            JOIN images ON detections.image_id = images.id
            WHERE batch_id = ?
            ORDER BY images.original_path, detections.created_at
        `);
        // We need to handle column name collisions if any. 
        // detections.id vs images.id.
        // SQLite returns both. JS driver might overwrite.
        // We should select explicit columns to avoid ID collision.
        // detections.id as detection_id, images.id as image_id (which matches DBImage id)
        
        const safeStmt = db.prepare(`
            SELECT 
                detections.id as detection_id,
                detections.batch_id,
                detections.image_id,
                detections.label,
                detections.confidence,
                detections.detection_confidence,
                detections.x1, detections.y1, detections.x2, detections.y2,
                detections.source,
                detections.created_at as detection_created_at,
                images.id as id, -- DBImage expects 'id' to be the image ID
                images.group_id,
                images.original_path,
                images.preview_path,
                images.date_added
            FROM detections
            JOIN images ON detections.image_id = images.id
            WHERE batch_id = ?
            ORDER BY images.original_path, detections.created_at
        `);
        
        return safeStmt.all(batchId) as (Detection & Image)[];
    },

    updateDetectionLabel: (id: number, label: string): void => {
        const stmt = db.prepare('UPDATE detections SET label = ? WHERE id = ?');
        stmt.run(label, id);
    },

    deleteDetection: (id: number): void => {
        const stmt = db.prepare('DELETE FROM detections WHERE id = ?');
        stmt.run(id);
    },

    // --- Cleanup ---

    cleanupMissingImages: (): number => {
        const images = db.prepare('SELECT id, original_path FROM images').all() as { id: number, original_path: string }[];
        let deletedCount = 0;
        const deleteStmt = db.prepare('DELETE FROM images WHERE id = ?');

        const deleteTransaction = db.transaction((idsToDelete: number[]) => {
            for (const id of idsToDelete) {
                deleteStmt.run(id);
            }
        });

        const idsToDelete: number[] = [];

        for (const img of images) {
            if (!fs.existsSync(img.original_path)) {
                idsToDelete.push(img.id);
                deletedCount++;
            }
        }

        if (idsToDelete.length > 0) {
            deleteTransaction(idsToDelete);
        }

        // Also cleanup empty groups? User didn't specify, but it's good practice.
        // Let's leave empty groups for now as user might want to keep them.

        return deletedCount;
    }
};
