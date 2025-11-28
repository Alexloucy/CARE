import {
    Button,
    Menu, MenuItem,
    useTheme
} from '@mui/material';
import {
    PencilSimple,
    Plus,
    Trash
} from '@phosphor-icons/react';
import React, { useEffect, useMemo, useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import { DBImage } from '../../types/electron';

// Hooks
import { useGroupActions } from '../../hooks/useGroupActions';
import { useImageLoader } from '../../hooks/useImageLoader';
import { useLibraryData } from '../../hooks/useLibraryData';
import { useLibraryUpload } from '../../hooks/useLibraryUpload';
import { useSelection } from '../../hooks/useSelection';

// Components
import { GroupNameDialog } from '../../components/GroupNameDialog';
import { LibraryFilter } from '../../components/library/LibraryFilterDialog';
import { MediaExplorer } from '../../components/library/MediaExplorer';

const LibraryPage: React.FC = () => {
    const theme = useTheme();
    const { leftSidebarOpen, rightSidebarOpen } = useOutletContext<{ leftSidebarOpen: boolean; rightSidebarOpen: boolean }>();

    // 1. Filter & Search State (Must be defined before data loading)
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
    // Fetch Full Library (for Filter Dialog metadata)
    const { dateSections: fullDateSections, refreshLibrary: refreshFullLibrary } = useLibraryData();
    
    // Fetch Filtered Library (for View)
    const { dateSections: filteredDateSections, loading, refreshLibrary: refreshFilteredLibrary } = useLibraryData(dbFilter);

    // Available species for ReID
    const [availableSpecies, setAvailableSpecies] = useState<string[]>([]);
    useEffect(() => {
        window.api.getAvailableSpecies().then(result => {
            if (result.ok && result.species) {
                setAvailableSpecies(result.species);
            }
        });
    }, []);

    // Unified Refresh
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

    // 5. Upload Logic
    const { 
        groupNameDialogOpen, 
        setGroupNameDialogOpen, 
        setPendingUploadFiles, 
        handleUploadClick, 
        processUploadPaths, 
        handleConfirmUpload 
    } = useLibraryUpload();

    // 5.1 Drag & Drop State
    const [isDragging, setIsDragging] = useState(false);

    // 6. Group Actions
    const {
        anchorEl,
        renameDialogOpen,
        groupToRename,
        setRenameDialogOpen,
        setGroupToRename,
        handleMenuOpen,
        handleMenuClose,
        handleDeleteGroup,
        handleRenameGroupClick,
        handleConfirmRename
    } = useGroupActions(refreshLibrary, fullDateSections);

    // Derived State
    const allImages = useMemo(() => {
        return filteredDateSections.flatMap(section => section.groups.flatMap(group => group.images));
    }, [filteredDateSections]);

    // Effect: Prune selection when filter hides items
    useEffect(() => {
        if (selectedImageIds.size === 0) return;

        const visibleIds = new Set(allImages.map(img => img.id));
        const nextSelection = new Set([...selectedImageIds].filter(id => visibleIds.has(id)));

        if (nextSelection.size !== selectedImageIds.size) {
            setSelection(nextSelection);
        }
    }, [allImages, selectedImageIds, setSelection]);

    // Batch Actions
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
            } else if (result.error !== 'Operation canceled') {
                alert(`Save failed: ${result.error}`);
            }
        } catch (error) {
            console.error('Batch save error:', error);
        }
    };

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
            alert('Failed to start classification: ' + error);
        }
    };

    const handleBatchReID = async (species: string) => {
        if (selectedImageIds.size === 0) return;
        const imageIds = Array.from(selectedImageIds);

        try {
            const result = await window.api.smartReID(imageIds, species);
            if (result.ok) {
                setIsSelectionMode(false);
                clearSelection();
            } else {
                alert('ReID failed: ' + result.error);
            }
        } catch (error) {
            console.error('Batch ReID error:', error);
            alert('Failed to start ReID: ' + error);
        }
    };

    const handleDeleteImage = async (image: DBImage) => {
        await window.api.deleteImage(image.id);
        await refreshLibrary();
    };

    // Group-level handlers for Analyse menu
    const handleGroupClassify = async (images: DBImage[]) => {
        const paths = images.map(img => img.original_path);
        try {
            await window.api.detect(paths, (txt) => console.log(txt));
        } catch (error) {
            console.error('Classification error:', error);
            alert('Failed to start classification: ' + error);
        }
    };

    const handleGroupReID = async (images: DBImage[], species: string) => {
        const imageIds = images.map(img => img.id);
        try {
            const result = await window.api.smartReID(imageIds, species);
            if (!result.ok) {
                alert('ReID failed: ' + result.error);
            }
        } catch (error) {
            console.error('ReID error:', error);
            alert('Failed to start ReID: ' + error);
        }
    };

    // Drag Drop Handlers
    const handleDrop = async (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const files = Array.from(e.dataTransfer.files);
        if (files.length === 0) return;
        const paths = files.map(file => window.api.getPathForFile(file));
        processUploadPaths(paths);
    };

    return (
        <>
            <MediaExplorer
                title="Library"
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
                onBatchReID={handleBatchReID}
                onBatchSave={handleBatchSave}
                availableSpecies={availableSpecies}
                aiButtonMode="analyse"
                onClassify={handleGroupClassify}
                onReID={handleGroupReID}
                onDeleteImage={handleDeleteImage}
                onDrop={handleDrop}
                isDragging={isDragging}
                setIsDragging={setIsDragging}
                leftSidebarOpen={leftSidebarOpen}
                rightSidebarOpen={rightSidebarOpen}
                onUpload={handleUploadClick}
                headerActions={
                    <Button variant="contained" startIcon={<Plus />} onClick={handleUploadClick} sx={{ borderRadius: 2, textTransform: 'none', px: 3 }}>
                        Upload
                    </Button>
                }
                onGroupMenuOpen={handleMenuOpen}
                groupMenu={
                    <Menu
                        anchorEl={anchorEl}
                        open={Boolean(anchorEl)}
                        onClose={handleMenuClose}
                        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
                        PaperProps={{
                            elevation: 0,
                            sx: {
                                backgroundColor: theme.palette.mode === 'light' ? 'rgba(255, 255, 255, 0.85)' : 'rgba(45, 45, 45, 0.85)',
                                backdropFilter: 'blur(8px)',
                                borderRadius: '8px',
                                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
                                border: `1px solid ${theme.palette.divider}`,
                                minWidth: '160px',
                                mt: 0.5
                            }
                        }}
                        MenuListProps={{ sx: { padding: '6px' } }}
                    >
                        <MenuItem onClick={handleRenameGroupClick} sx={{ borderRadius: '6px', margin: '2px 0', gap: 1 }}>
                            <PencilSimple size={18} /> Rename
                        </MenuItem>
                        <MenuItem onClick={handleDeleteGroup} sx={{ borderRadius: '6px', margin: '2px 0', gap: 1, color: 'error.main' }}>
                            <Trash size={18} /> Delete
                        </MenuItem>
                    </Menu>
                }
            />

            {/* Dialogs - Kept in Page because they are business logic specific (Renaming Group) */}
            <GroupNameDialog
                open={groupNameDialogOpen}
                onClose={() => { setGroupNameDialogOpen(false); setPendingUploadFiles([]); }}
                onConfirm={handleConfirmUpload}
                title="Create New Group"
            />

            <GroupNameDialog
                open={renameDialogOpen}
                onClose={() => { setRenameDialogOpen(false); setGroupToRename(null); }}
                onConfirm={handleConfirmRename}
                title="Rename Group"
                initialValue={groupToRename?.name || ''}
            />
        </>
    );
};

export default LibraryPage;
