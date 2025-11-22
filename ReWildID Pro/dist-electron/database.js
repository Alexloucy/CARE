"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DatabaseService = void 0;
const better_sqlite3_1 = __importDefault(require("better-sqlite3"));
const path_1 = __importDefault(require("path"));
const fs_1 = __importDefault(require("fs"));
const isDev = process.env.NODE_ENV === 'development';
// Determine database path
// In production, we might want to store it in appData, but for now adhering to process.cwd()/data as per previous logic
const DATA_DIR = path_1.default.join(process.cwd(), 'data');
const DB_PATH = path_1.default.join(DATA_DIR, 'library.db');
if (!fs_1.default.existsSync(DATA_DIR)) {
    fs_1.default.mkdirSync(DATA_DIR, { recursive: true });
}
const db = new better_sqlite3_1.default(DB_PATH, { verbose: isDev ? console.log : undefined });
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
exports.DatabaseService = {
    // --- Groups ---
    createGroup: (name, createdAt) => {
        const stmt = db.prepare('INSERT INTO groups (name, created_at, updated_at) VALUES (?, ?, ?)');
        const now = Date.now();
        const info = stmt.run(name, createdAt || now, now);
        return info.lastInsertRowid;
    },
    getGroup: (id) => {
        const stmt = db.prepare('SELECT * FROM groups WHERE id = ?');
        return stmt.get(id);
    },
    updateGroupName: (id, name) => {
        const stmt = db.prepare('UPDATE groups SET name = ?, updated_at = ? WHERE id = ?');
        stmt.run(name, Date.now(), id);
    },
    deleteGroup: (id) => {
        const stmt = db.prepare('DELETE FROM groups WHERE id = ?');
        stmt.run(id);
    },
    getAllGroups: () => {
        const stmt = db.prepare('SELECT * FROM groups ORDER BY created_at DESC');
        return stmt.all();
    },
    // --- Images ---
    addImage: (groupId, originalPath, previewPath) => {
        const stmt = db.prepare('INSERT INTO images (group_id, original_path, preview_path, date_added) VALUES (?, ?, ?, ?)');
        const info = stmt.run(groupId, originalPath, previewPath || null, Date.now());
        return info.lastInsertRowid;
    },
    deleteImage: (id) => {
        const stmt = db.prepare('DELETE FROM images WHERE id = ?');
        stmt.run(id);
    },
    getImages: () => {
        // Cleanup missing files first (optional, but requested behavior)
        // We can do this async or periodically, but for now let's do it on fetch to ensure consistency
        // However, scanning all files might be slow. Let's do a quick check or separate method.
        // For now, just return data. Cleanup should be explicit or background.
        const stmt = db.prepare(`
            SELECT images.*, groups.name as group_name, groups.created_at as group_created_at
            FROM images
            JOIN groups ON images.group_id = groups.id
            ORDER BY groups.created_at DESC, images.date_added DESC
        `);
        return stmt.all();
    },
    // --- Cleanup ---
    cleanupMissingImages: () => {
        const images = db.prepare('SELECT id, original_path FROM images').all();
        let deletedCount = 0;
        const deleteStmt = db.prepare('DELETE FROM images WHERE id = ?');
        const deleteTransaction = db.transaction((idsToDelete) => {
            for (const id of idsToDelete) {
                deleteStmt.run(id);
            }
        });
        const idsToDelete = [];
        for (const img of images) {
            if (!fs_1.default.existsSync(img.original_path)) {
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
