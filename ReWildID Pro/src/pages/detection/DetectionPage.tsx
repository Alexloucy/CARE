import React, { useMemo, useState, useEffect } from 'react';
import { useOutletContext } from 'react-router-dom';
import { DBImage } from '../../types/electron';

// Hooks
import { useImageLoader } from '../../hooks/useImageLoader';
// import { useLibraryData } from '../../hooks/useLibraryData'; // Replaced with custom loading
import { useSelection } from '../../hooks/useSelection';

// Components
import { LibraryFilter } from '../../components/library/LibraryFilterDialog';
import { MediaExplorer } from '../../components/library/MediaExplorer';
import { DateSection, GroupData } from '../../types/library';

const DetectionPage: React.FC = () => {
    const { leftSidebarOpen, rightSidebarOpen } = useOutletContext<{ leftSidebarOpen: boolean; rightSidebarOpen: boolean }>();

    // 1. Filter & Search State
    const [filterDialogOpen, setFilterDialogOpen] = useState(false);
    const [activeFilter, setActiveFilter] = useState<LibraryFilter | null>(null);
    const [searchQuery, setSearchQuery] = useState('');

    // 2. Data & Loading
    const [loading, setLoading] = useState(false);
    const [filteredDateSections, setFilteredDateSections] = useState<DateSection[]>([]);
    const [fullDateSections, setFullDateSections] = useState<DateSection[]>([]); // Needed for timeline?
    const [refreshTrigger, setRefreshTrigger] = useState(0);

    const refreshLibrary = async () => {
        setRefreshTrigger(prev => prev + 1);
    };
    
    // Data Loading Effect
    useEffect(() => {
        const loadData = async () => {
            setLoading(true);
            try {
                const batchesRes = await window.api.getDetectionBatches();
                if (batchesRes.ok && batchesRes.batches) {
                    // Sort batches by date descending
                    const sortedBatches = batchesRes.batches.sort((a, b) => b.created_at - a.created_at);
                    
                    const sectionsMap = new Map<string, GroupData[]>(); // DateKey -> Groups

                    for (const batch of sortedBatches) {
                        const detRes = await window.api.getDetectionsForBatch(batch.id);
                        if (detRes.ok && detRes.detections) {
                            // Group detections by Image ID
                            const imagesMap = new Map<number, DBImage>();
                            
                            for (const d of detRes.detections) {
                                // d is (Detection & Image) from backend query
                                const imageId = (d as any).id || d.image_id; // In my query I aliased images.id as 'id'
                                
                                if (!imagesMap.has(imageId)) {
                                    imagesMap.set(imageId, {
                                        id: imageId,
                                        group_id: 0, // Placeholder
                                        original_path: (d as any).original_path || '',
                                        preview_path: (d as any).preview_path,
                                        date_added: (d as any).date_added,
                                        group_name: batch.name,
                                        group_created_at: batch.created_at,
                                        detections: []
                                    });
                                }
                                
                                // Add detection info
                                imagesMap.get(imageId)?.detections?.push({
                                    id: (d as any).detection_id,
                                    label: d.label,
                                    confidence: d.confidence,
                                    detection_confidence: d.detection_confidence,
                                    x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2,
                                    source: d.source,
                                    created_at: (d as any).detection_created_at,
                                    batch_id: d.batch_id,
                                    image_id: d.image_id
                                });
                            }
                            
                            const images = Array.from(imagesMap.values());
                            
                            if (images.length > 0) {
                                // Group by Date
                                const dateObj = new Date(batch.created_at);
                                const y = dateObj.getFullYear();
                                const m = String(dateObj.getMonth() + 1).padStart(2, '0');
                                const day = String(dateObj.getDate()).padStart(2, '0');
                                const dateKey = `${y}${m}${day}`;
                                
                                if (!sectionsMap.has(dateKey)) sectionsMap.set(dateKey, []);
                                
                                sectionsMap.get(dateKey)?.push({
                                    id: batch.id,
                                    name: batch.name,
                                    created_at: batch.created_at,
                                    images: images
                                });
                            }
                        }
                    }
                    
                    // Convert to Sections
                    const newSections: DateSection[] = [];
                    // Sort dates descending
                    const sortedDates = Array.from(sectionsMap.keys()).sort((a, b) => b.localeCompare(a));
                    
                    for (const dateKey of sortedDates) {
                        newSections.push({
                            date: dateKey,
                            groups: sectionsMap.get(dateKey) || []
                        });
                    }
                    
                    setFilteredDateSections(newSections);
                    setFullDateSections(newSections); // For now, sync them
                }
            } catch (e) {
                console.error("Failed to load detections:", e);
            } finally {
                setLoading(false);
            }
        };
        loadData();
    }, [refreshTrigger]);

    // 3. Image Loading
    const { imageUrls, fullImageUrls, loadImage, loadFullImage } = useImageLoader();

    // 4. Selection
    const { 
        isSelectionMode, 
        selectedIds: selectedImageIds, 
        toggleSelectionMode, 
        toggleItem: toggleImageSelection, 
        clearSelection,
        setIsSelectionMode,
        setSelection
    } = useSelection<number>();

    // Derived State
    const allImages = useMemo(() => {
        return filteredDateSections.flatMap(section => section.groups.flatMap(group => group.images));
    }, [filteredDateSections]);

    // Batch Actions
    const handleBatchDelete = async () => {
        if (selectedImageIds.size === 0) return;
        if (window.confirm(`Are you sure you want to delete ${selectedImageIds.size} images?`)) {
            try {
                for (const id of selectedImageIds) {
                    await window.api.deleteImage(id);
                }
                await refreshLibrary();
                clearSelection();
            } catch (error) {
                console.error('Batch delete error:', error);
            }
        }
    };

    const handleBatchDetect = async () => {
        if (selectedImageIds.size === 0) return;
        const paths: string[] = [];
        allImages.forEach(img => {
            if (selectedImageIds.has(img.id)) paths.push(img.original_path);
        });
        if (paths.length === 0) return;

        try {
            await window.api.detect(paths, (txt) => console.log(txt));
            setIsSelectionMode(false);
            clearSelection();
        } catch (error) {
            console.error('Batch detect error:', error);
            alert('Failed to start detection: ' + error);
        }
    };

    const handleBatchSave = async () => {
        if (selectedImageIds.size === 0) return;
        const paths: string[] = [];
        allImages.forEach(img => {
            if (selectedImageIds.has(img.id)) paths.push(img.original_path);
        });
        if (paths.length === 0) return;
        
        try {
             const result = await window.api.saveImages(paths);
             if (result.ok) {
                 alert(`Successfully saved ${result.successCount} images.`);
                 clearSelection();
             }
        } catch (e) {
            console.error(e);
        }
    };

    const handleDeleteImage = async (image: DBImage) => {
        await window.api.deleteImage(image.id);
        await refreshLibrary();
    };

    return (
        <MediaExplorer
            title="Detection"
            loading={loading}
            dateSections={filteredDateSections}
            fullDateSections={fullDateSections}
            imageUrls={imageUrls}
            fullImageUrls={fullImageUrls}
            allImages={allImages}
            loadImage={loadImage}
            loadFullImage={loadFullImage}
            activeFilter={activeFilter}
            onFilterChange={setActiveFilter}
            searchQuery={searchQuery}
            onSearchChange={setSearchQuery}
            filterDialogOpen={filterDialogOpen}
            setFilterDialogOpen={setFilterDialogOpen}
            isSelectionMode={isSelectionMode}
            selectedImageIds={selectedImageIds}
            toggleSelectionMode={toggleSelectionMode}
            toggleImageSelection={toggleImageSelection}
            setSelection={setSelection}
            clearSelection={clearSelection}
            setIsSelectionMode={setIsSelectionMode}
            onBatchDelete={handleBatchDelete}
            onBatchDetect={handleBatchDetect}
            onBatchSave={handleBatchSave}
            onDeleteImage={handleDeleteImage}
            leftSidebarOpen={leftSidebarOpen}
            rightSidebarOpen={rightSidebarOpen}
        />
    );
};

export default DetectionPage;
