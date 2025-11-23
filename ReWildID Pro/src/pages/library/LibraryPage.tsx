import React, { useEffect, useState, useMemo } from 'react';
import { Box, Typography, useTheme, Skeleton, IconButton, Button, Tooltip, Menu, MenuItem, Card } from '@mui/material';
import { Plus, PencilSimple, Trash, CheckSquare, X } from '@phosphor-icons/react';
import { DBImage } from '../../types/electron';

// Hooks
import { useLibraryData } from '../../hooks/useLibraryData';
import { useImageLoader } from '../../hooks/useImageLoader';
import { useSelection } from '../../hooks/useSelection';
import { useLibraryUpload } from '../../hooks/useLibraryUpload';
import { useGroupActions } from '../../hooks/useGroupActions';

// Components
import ImageModal from '../../components/ImageModal';
import { GroupNameDialog } from '../../components/GroupNameDialog';
import { DragDropOverlay } from '../../components/library/DragDropOverlay';
import { SelectionToolbar } from '../../components/library/SelectionToolbar';
import { DateGroupList } from '../../components/library/DateGroupList';

const LibraryPage: React.FC = () => {
    const theme = useTheme();

    // 1. Data & Loading
    const { dateSections, loading, refreshLibrary } = useLibraryData();
    
    // 2. Image Loading
    const { imageUrls, fullImageUrls, loadImage, loadFullImage } = useImageLoader();

    // 3. Selection
    const { 
        isSelectionMode, 
        selectedIds: selectedImageIds, 
        toggleSelectionMode, 
        toggleItem: toggleImageSelection, 
        clearSelection
    } = useSelection<number>();

    // 4. Upload Logic
    const { 
        groupNameDialogOpen, 
        setGroupNameDialogOpen, 
        setPendingUploadFiles, 
        handleUploadClick, 
        processUploadPaths, 
        handleConfirmUpload 
    } = useLibraryUpload(refreshLibrary);

    // 5. Group Actions
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
    } = useGroupActions(refreshLibrary, dateSections);

    // 6. Local State
    const [selectedImage, setSelectedImage] = useState<{ image: DBImage, url: string } | null>(null);
    const [isDragging, setIsDragging] = useState(false);

    // Derived State
    const allImages = useMemo(() => {
        return dateSections.flatMap(section => section.groups.flatMap(group => group.images));
    }, [dateSections]);

    // Effect: Load full image when modal opens
    useEffect(() => {
        if (selectedImage) {
            loadFullImage(selectedImage.image);
        }
    }, [selectedImage?.image.id, loadFullImage]);

    // Effect: Update modal URL when full image loads
    useEffect(() => {
        if (selectedImage) {
            if (fullImageUrls[selectedImage.image.id]) {
                setSelectedImage(prev => prev ? { ...prev, url: fullImageUrls[selectedImage.image.id] } : null);
            }
        }
    }, [fullImageUrls, selectedImage?.image.id]);

    // Navigation Handlers
    const handleNext = () => {
        if (!selectedImage) return;
        const currentIndex = allImages.findIndex(img => img.id === selectedImage.image.id);
        if (currentIndex < allImages.length - 1) {
            const nextImage = allImages[currentIndex + 1];
            if (!imageUrls[nextImage.id]) loadImage(nextImage);
            setSelectedImage({ image: nextImage, url: imageUrls[nextImage.id] || '' });
        }
    };

    const handlePrev = () => {
        if (!selectedImage) return;
        const currentIndex = allImages.findIndex(img => img.id === selectedImage.image.id);
        if (currentIndex > 0) {
            const prevImage = allImages[currentIndex - 1];
            if (!imageUrls[prevImage.id]) loadImage(prevImage);
            setSelectedImage({ image: prevImage, url: fullImageUrls[prevImage.id] || imageUrls[prevImage.id] || '' });
        }
    };

    // Drag Drop Handlers
    const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
    const handleDragLeave = (e: React.DragEvent) => { 
        e.preventDefault(); 
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setIsDragging(false); 
    };
    const handleDrop = async (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const files = Array.from(e.dataTransfer.files);
        if (files.length === 0) return;
        const paths = files.map(file => window.api.getPathForFile(file));
        processUploadPaths(paths);
    };

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

    const handleDeleteImage = async () => {
        if (!selectedImage) return;
        await window.api.deleteImage(selectedImage.image.id);
        setSelectedImage(null);
        await refreshLibrary();
    };

    return (
        <Box
            sx={{ height: '100%', position: 'relative', outline: 'none', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
            <DragDropOverlay isDragging={isDragging} />

            {/* Header */}
            <Box sx={{ p: 3, px: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: theme.palette.background.default, zIndex: 10 }}>
                <Typography variant="h4" fontWeight="bold">Library</Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                    <Tooltip title={isSelectionMode ? "Cancel Selection" : "Select Items"}>
                        <IconButton
                            onClick={toggleSelectionMode}
                            color={isSelectionMode ? "primary" : "default"}
                            sx={{
                                bgcolor: isSelectionMode ? (theme.palette.mode === 'light' ? 'rgba(25, 118, 210, 0.08)' : 'rgba(144, 202, 249, 0.16)') : 'transparent',
                                '&:hover': { bgcolor: isSelectionMode ? (theme.palette.mode === 'light' ? 'rgba(25, 118, 210, 0.12)' : 'rgba(144, 202, 249, 0.24)') : theme.palette.action.hover }
                            }}
                        >
                            {isSelectionMode ? <X weight="bold" /> : <CheckSquare weight="regular" />}
                        </IconButton>
                    </Tooltip>
                    <Button variant="contained" startIcon={<Plus />} onClick={handleUploadClick} sx={{ borderRadius: 2, textTransform: 'none', px: 3 }}>
                        Upload
                    </Button>
                </Box>
            </Box>

            {/* Content */}
            <Box sx={{ flex: 1, overflowY: 'auto', p: 4, pt: 0 }}>
                {loading ? (
                    <Box sx={{ mt: 2 }}>
                        {/* Date Header Skeleton */}
                        <Skeleton variant="text" sx={{ fontSize: '0.875rem', width: 200, mb: 2 }} />

                        {/* Group Header Skeleton */}
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                            <Skeleton variant="text" sx={{ fontSize: '1.25rem', width: 150 }} />
                            <Skeleton variant="rounded" width={40} height={24} />
                        </Box>

                        {/* Image Grid Skeleton */}
                        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 2 }}>
                            {[...Array(12)].map((_, i) => (
                                <Card key={i} sx={{ aspectRatio: '1/1', width: '100%', borderRadius: 3, boxShadow: 'none' }}>
                                    <Skeleton variant="rectangular" width="100%" height="100%" animation="wave" />
                                </Card>
                            ))}
                        </Box>
                    </Box>
                ) : (
                    <DateGroupList 
                        dateSections={dateSections}
                        imageUrls={imageUrls}
                        loadImage={loadImage}
                        isSelectionMode={isSelectionMode}
                        selectedImageIds={selectedImageIds}
                        onToggleSelection={toggleImageSelection}
                        onImageClick={(img) => {
                            if (isSelectionMode) toggleImageSelection(img.id);
                            else if (imageUrls[img.id]) setSelectedImage({ image: img, url: imageUrls[img.id] });
                        }}
                        onMenuOpen={handleMenuOpen}
                    />
                )}
            </Box>

            {/* Selection Toolbar */}
            <SelectionToolbar 
                visible={isSelectionMode && selectedImageIds.size > 0}
                selectedCount={selectedImageIds.size}
                onSave={handleBatchSave}
                onDelete={handleBatchDelete}
            />

            {/* Image Modal */}
            <ImageModal
                open={!!selectedImage}
                onClose={() => setSelectedImage(null)}
                imageUrl={selectedImage?.url}
                file={selectedImage ? {
                    name: selectedImage.image.original_path.split(/[\\/]/).pop() || 'image.jpg',
                    path: selectedImage.image.original_path,
                    isDirectory: false
                } : undefined}
                onNext={handleNext}
                onPrev={handlePrev}
                hasNext={selectedImage ? allImages.findIndex(img => img.id === selectedImage.image.id) < allImages.length - 1 : false}
                hasPrev={selectedImage ? allImages.findIndex(img => img.id === selectedImage.image.id) > 0 : false}
                onDelete={handleDeleteImage}
            />

            {/* Dialogs */}
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

            {/* Group Menu */}
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
        </Box>
    );
};

export default LibraryPage;
