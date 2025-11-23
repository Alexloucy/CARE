import { useState, useCallback } from 'react';
import { DBImage } from '../types/electron';

export function useImageLoader() {
    const [imageUrls, setImageUrls] = useState<Record<number, string>>({}); // Map ID -> URL (Thumbnails)
    const [fullImageUrls, setFullImageUrls] = useState<Record<number, string>>({}); // Map ID -> URL (Full Res)

    const loadFullImage = useCallback(async (image: DBImage) => {
        if (fullImageUrls[image.id]) return;

        try {
            const response = await window.api.viewImage(image.original_path);
            
            if (response.ok && response.data) {
                const blob = new Blob([response.data as unknown as BlobPart], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                setFullImageUrls(prev => ({ ...prev, [image.id]: url }));
            }
        } catch (error) {
            console.error(`Failed to load full image ${image.original_path}:`, error);
        }
    }, [fullImageUrls]);

    const loadImage = useCallback(async (image: DBImage) => {
        if (imageUrls[image.id]) return;

        try {
            // Prioritize preview path if available
            const pathToCheck = image.preview_path || image.original_path;
            const response = await window.api.viewImage(pathToCheck);
            
            if (response.ok && response.data) {
                const blob = new Blob([response.data as unknown as BlobPart], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                setImageUrls(prev => ({ ...prev, [image.id]: url }));
            }
        } catch (error) {
            console.error(`Failed to load image ${image.original_path}:`, error);
        }
    }, [imageUrls]);

    return { imageUrls, fullImageUrls, loadImage, loadFullImage };
}
