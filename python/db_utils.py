"""
Database utilities for embedding cache.

This module provides the EmbeddingCache class for storing and retrieving
DINOv3 embeddings from the SQLite database.
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class EmbeddingCache:
    """Cache for storing and retrieving DINOv3 embeddings."""
    
    def __init__(self, db_path: str):
        """
        Initialize the embedding cache.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_table()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_table(self):
        """Create embeddings table if it doesn't exist."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    bbox_hash TEXT NOT NULL,
                    embedding_type TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_lookup 
                ON embeddings(image_id, bbox_hash, embedding_type)
            """)
            conn.commit()
        finally:
            conn.close()
    
    @staticmethod
    def bbox_to_hash(bbox: List[float]) -> str:
        """
        Convert bbox [x1, y1, x2, y2] to hash string.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2].
            
        Returns:
            Hash string in format "x1_y1_x2_y2".
        """
        return f"{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"
    
    def get_embedding(
        self, 
        image_id: int, 
        bbox: List[float], 
        embedding_type: str
    ) -> Optional[np.ndarray]:
        """
        Get cached embedding for a specific crop.
        
        Args:
            image_id: Database image ID.
            bbox: Bounding box coordinates [x1, y1, x2, y2].
            embedding_type: Type of embedding (e.g., 'dinov3_raw').
            
        Returns:
            numpy array of embedding, or None if not found.
        """
        bbox_hash = self.bbox_to_hash(bbox)
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT embedding FROM embeddings 
                WHERE image_id = ? AND bbox_hash = ? AND embedding_type = ?
            """, (image_id, bbox_hash, embedding_type))
            row = cursor.fetchone()
            if row:
                return np.frombuffer(row['embedding'], dtype=np.float32)
            return None
        finally:
            conn.close()
    
    def store_embedding(
        self, 
        image_id: int, 
        bbox: List[float], 
        embedding_type: str, 
        embedding: np.ndarray
    ):
        """
        Store embedding in cache.
        
        Args:
            image_id: Database image ID.
            bbox: Bounding box coordinates [x1, y1, x2, y2].
            embedding_type: Type of embedding (e.g., 'dinov3_raw').
            embedding: numpy array of embedding.
        """
        bbox_hash = self.bbox_to_hash(bbox)
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings 
                (image_id, bbox_hash, embedding_type, embedding, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (image_id, bbox_hash, embedding_type, embedding_bytes, int(__import__('time').time() * 1000)))
            conn.commit()
        finally:
            conn.close()
    
    def get_embeddings_batch(
        self, 
        items: List[Tuple[int, List[float]]],  # [(image_id, bbox), ...]
        embedding_type: str
    ) -> Dict[str, np.ndarray]:
        """
        Batch lookup of embeddings.
        
        Args:
            items: List of (image_id, bbox) tuples.
            embedding_type: Type of embedding (e.g., 'dinov3_raw').
            
        Returns:
            Dict keyed by "image_id:bbox_hash" with numpy array values.
        """
        result = {}
        if not items:
            return result
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            for image_id, bbox in items:
                bbox_hash = self.bbox_to_hash(bbox)
                cursor.execute("""
                    SELECT embedding FROM embeddings 
                    WHERE image_id = ? AND bbox_hash = ? AND embedding_type = ?
                """, (image_id, bbox_hash, embedding_type))
                row = cursor.fetchone()
                if row:
                    key = f"{image_id}:{bbox_hash}"
                    result[key] = np.frombuffer(row['embedding'], dtype=np.float32)
            return result
        finally:
            conn.close()
    
    def store_embeddings_batch(
        self, 
        items: List[Tuple[int, List[float], np.ndarray]],  # [(image_id, bbox, embedding), ...]
        embedding_type: str
    ):
        """
        Batch store embeddings.
        
        Args:
            items: List of (image_id, bbox, embedding) tuples.
            embedding_type: Type of embedding (e.g., 'dinov3_raw').
        """
        if not items:
            return
            
        import time
        now = int(time.time() * 1000)
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            for image_id, bbox, embedding in items:
                bbox_hash = self.bbox_to_hash(bbox)
                embedding_bytes = embedding.astype(np.float32).tobytes()
                cursor.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (image_id, bbox_hash, embedding_type, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (image_id, bbox_hash, embedding_type, embedding_bytes, now))
            conn.commit()
        finally:
            conn.close()
    
    def count_embeddings(self, embedding_type: Optional[str] = None) -> int:
        """
        Count total embeddings in cache.
        
        Args:
            embedding_type: Optional type filter.
            
        Returns:
            Count of embeddings.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if embedding_type:
                cursor.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE embedding_type = ?",
                    (embedding_type,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM embeddings")
            return cursor.fetchone()[0]
        finally:
            conn.close()


# Convenience function for quick cache creation
def create_cache(db_path: str) -> EmbeddingCache:
    """Create an EmbeddingCache instance from database path."""
    return EmbeddingCache(db_path)
