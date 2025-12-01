import { useState, useEffect, useCallback } from 'react';
import { Detection } from '../types/electron';

export interface ReidInfo {
    individualId: number;
    individualName: string;
    individualDisplayName: string;
    individualColor: string;
    runId: number;
    runName: string;
    species: string;
}

export interface ImageMetadata {
    detections: Detection[];
    reidResults: ReidInfo[];
}

/**
 * Hook to fetch detection and ReID metadata for a set of image IDs.
 * Returns a map of imageId -> ImageMetadata.
 */
export function useImageMetadata(imageIds: number[]) {
    const [metadata, setMetadata] = useState<Record<number, ImageMetadata>>({});
    const [loading, setLoading] = useState(false);

    const fetchMetadata = useCallback(async () => {
        if (imageIds.length === 0) {
            setMetadata({});
            return;
        }

        setLoading(true);
        try {
            // Fetch detections and ReID results in parallel
            const [detectionsResult, reidResult] = await Promise.all([
                window.api.getLatestDetectionsForImages(imageIds),
                window.api.getReidResultsForImages(imageIds)
            ]);

            const newMetadata: Record<number, ImageMetadata> = {};

            // Initialize all image IDs with empty arrays
            imageIds.forEach(id => {
                newMetadata[id] = { detections: [], reidResults: [] };
            });

            // Populate detections
            if (detectionsResult.ok && detectionsResult.detections) {
                detectionsResult.detections.forEach(det => {
                    if (!newMetadata[det.image_id]) {
                        newMetadata[det.image_id] = { detections: [], reidResults: [] };
                    }
                    newMetadata[det.image_id].detections.push(det);
                });
            }

            // Populate ReID results
            if (reidResult.ok && reidResult.results) {
                reidResult.results.forEach(reid => {
                    if (!newMetadata[reid.imageId]) {
                        newMetadata[reid.imageId] = { detections: [], reidResults: [] };
                    }
                    // Avoid duplicates (same individual might appear multiple times per image)
                    const existing = newMetadata[reid.imageId].reidResults.find(
                        r => r.individualId === reid.individualId
                    );
                    if (!existing) {
                        newMetadata[reid.imageId].reidResults.push({
                            individualId: reid.individualId,
                            individualName: reid.individualName,
                            individualDisplayName: reid.individualDisplayName,
                            individualColor: reid.individualColor,
                            runId: reid.runId,
                            runName: reid.runName,
                            species: reid.species
                        });
                    }
                });
            }

            setMetadata(newMetadata);
        } catch (error) {
            console.error('Error fetching image metadata:', error);
        } finally {
            setLoading(false);
        }
    }, [imageIds.join(',')]); // Using join to create a stable dependency

    useEffect(() => {
        fetchMetadata();
    }, [fetchMetadata]);

    return { metadata, loading, refresh: fetchMetadata };
}
