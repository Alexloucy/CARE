import React, { useMemo, useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import { DBImage } from '../../types/electron';

// Hooks
import { useImageLoader } from '../../hooks/useImageLoader';
import { useLibraryData } from '../../hooks/useLibraryData';
import { useSelection } from '../../hooks/useSelection';

// Components
import { LibraryFilter } from '../../components/library/LibraryFilterDialog';
import { MediaExplorer } from '../../components/library/MediaExplorer';

const DetectionPage: React.FC = () => {
    const { leftSidebarOpen, rightSidebarOpen } = useOutletContext<{ leftSidebarOpen: boolean; rightSidebarOpen: boolean }>();

    // 1. Filter & Search State
    const [filterDialogOpen, setFilterDialogOpen] = useState(false);
    const [activeFilter, setActiveFilter] = useState<LibraryFilter | null>(null);
    const [searchQuery, setSearchQuery] = useState('');

    // Construct DB Filter
    const dbFilter = useMemo(() => ({
        date: activeFilter?.date || null,
        groupIds: activeFilter?.groupIds || null,
        searchQuery: searchQuery || undefined
    }), [activeFilter, searchQuery]);

    // 2. Data & Loading
    const { dateSections: fullDateSections, refreshLibrary: refreshFullLibrary } = useLibraryData();
    const { dateSections: filteredDateSections, loading, refreshLibrary: refreshFilteredLibrary } = useLibraryData(dbFilter);

    const refreshLibrary = async () => {
        await Promise.all([refreshFullLibrary(), refreshFilteredLibrary()]);
    };
    
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
