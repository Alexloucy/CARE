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

    db.exec(createGroupsTable);
    db.exec(createImagesTable);
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
