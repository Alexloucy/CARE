import {
    Box,
    Button,
    Card, GlobalStyles,
    IconButton,
    Menu, MenuItem,
    Skeleton,
    Slider,
    Tooltip,
    Typography, useTheme,
    Switch, Divider,
} from '@mui/material';
import {
    CheckSquare,
    Gear,
    PencilSimple,
    Plus,
    Trash,
    X,
    ArrowCounterClockwise,
    Funnel
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
import ImageModal from '../../components/ImageModal';
import { DateGroupList } from '../../components/library/DateGroupList';
import { DragDropOverlay } from '../../components/library/DragDropOverlay';
import { Timeline } from '../../components/library/Timeline';
import { LibraryFilter, LibraryFilterDialog } from '../../components/library/LibraryFilterDialog';
import { LibrarySearchBar } from '../../components/library/LibrarySearchBar';
import { LibrarySelectionBar } from '../../components/library/LibrarySelectionBar';

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

    // 7. Local State
    const [selectedImage, setSelectedImage] = useState<{ image: DBImage, url: string } | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [activeId, setActiveId] = useState<string>('');
    const [gridItemSize, setGridItemSize] = useState(180);
    const [showFileNames, setShowFileNames] = useState(false);
    const [settingsAnchorEl, setSettingsAnchorEl] = useState<null | HTMLElement>(null);
    
    // Zoom Handler
    const handleWheel = (e: React.WheelEvent) => {
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            const delta = e.deltaY * -2.5; // Increased sensitivity
            setGridItemSize(prev => {
                const newVal = prev + delta;
                return Math.min(Math.max(newVal, 100), 400); // Clamp between 100 and 400
            });
        }
    };

    // Reset View
    const handleResetView = () => {
        setGridItemSize(180);
        setShowFileNames(false);
    };

    // Scroll Observer
    useEffect(() => {
        if (loading || filteredDateSections.length === 0) return;

        const observer = new IntersectionObserver((entries) => {
            // Filter intersecting entries and sort by top position to find the topmost one
            const visibleEntries = entries
                .filter(entry => entry.isIntersecting)
                .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);

            if (visibleEntries.length > 0) {
                // The first one is the topmost visible element
                setActiveId(visibleEntries[0].target.id);
            }
        }, {
            rootMargin: '0px 0px -80% 0px', // Start from top, active zone is top 20%
            threshold: 0
        });

        // Observe groups only (skipping dates as they are too short)
        filteredDateSections.forEach(section => {
            // const dateEl = document.getElementById(`date-${section.date}`);
            // if (dateEl) observer.observe(dateEl);
            
            section.groups.forEach(group => {
                const groupEl = document.getElementById(`group-${group.id}`);
                if (groupEl) observer.observe(groupEl);
            });
        });

        return () => observer.disconnect();
    }, [loading, filteredDateSections]);

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

    const handleDateClick = (date: string) => {
        const element = document.getElementById(`date-${date}`);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth' });
        }
    };

    const handleGroupClick = (groupId: number) => {
        const element = document.getElementById(`group-${groupId}`);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth' });
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
            alert('Failed to start detection: ' + error);
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
            sx={{ 
                height: '100%', 
                position: 'relative', 
                outline: 'none', 
                overflow: 'hidden', 
                display: 'flex', 
                flexDirection: 'column',
            }}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
            <GlobalStyles styles={{
                '*::-webkit-scrollbar': {
                  width: '0px',
                  height: '0px',
                  display: 'none'
                },
                '*': {
                    scrollbarWidth: 'none',
                    '-ms-overflow-style': 'none',
                }
            }} />
            <DragDropOverlay isDragging={isDragging} />
            
            {!loading && filteredDateSections.length > 0 && (
                <Timeline 
                    dateSections={filteredDateSections}
                    onDateClick={handleDateClick}
                    onGroupClick={handleGroupClick}
                    activeId={activeId}
                />
            )}

            {/* Header */}
            <Box sx={{ p: 3, px: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: theme.palette.background.default, zIndex: 10 }}>
                <Typography variant="h4" fontWeight="bold">Library</Typography>
                <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                    <LibrarySearchBar onSearch={setSearchQuery} />
                    <Tooltip title="Filter Library">
                        <IconButton 
                            onClick={() => setFilterDialogOpen(true)}
                            color={activeFilter ? 'inherit' : 'default'}
                            sx={{ 
                                bgcolor: activeFilter ? (theme.palette.mode === 'light' ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.12)') : 'transparent',
                                '&:hover': { bgcolor: activeFilter ? (theme.palette.mode === 'light' ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.20)') : theme.palette.action.hover }
                            }}
                        >
                            <Funnel weight={activeFilter ? "fill" : "regular"} />
                        </IconButton>
                    </Tooltip>
                    <Tooltip title="View Settings">
                        <IconButton
                            onClick={(e) => setSettingsAnchorEl(e.currentTarget)}
                            sx={{
                                '&:hover': { bgcolor: theme.palette.action.hover }
                            }}
                        >
                            <Gear weight="regular" />
                        </IconButton>
                    </Tooltip>
                    <Tooltip title={isSelectionMode ? "Cancel Selection" : "Select Items"}>
                        <IconButton
                            onClick={toggleSelectionMode}
                            color={isSelectionMode ? "inherit" : "default"}
                            sx={{
                                bgcolor: isSelectionMode ? (theme.palette.mode === 'light' ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.12)') : 'transparent',
                                '&:hover': { bgcolor: isSelectionMode ? (theme.palette.mode === 'light' ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.20)') : theme.palette.action.hover }
                            }}
                        >
                            {isSelectionMode ? <X weight="bold" /> : <CheckSquare weight={isSelectionMode ? "fill" : "regular"} />}
                        </IconButton>
                    </Tooltip>
                    <Button variant="contained" startIcon={<Plus />} onClick={handleUploadClick} sx={{ borderRadius: 2, textTransform: 'none', px: 3 }}>
                        Upload
                    </Button>
                </Box>
            </Box>

            {/* Content */}
            <Box 
                sx={{ 
                    flex: 1, 
                    overflowY: 'auto', 
                    p: 4, 
                    pt: 0,
                    '&::-webkit-scrollbar': { display: 'none' },
                    scrollbarWidth: 'none',
                }}
                onWheel={handleWheel}
            >
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
                        dateSections={filteredDateSections}
                        imageUrls={imageUrls}
                        loadImage={loadImage}
                        isSelectionMode={isSelectionMode}
                        selectedImageIds={selectedImageIds}
                        onToggleSelection={toggleImageSelection}
                        onSetSelection={setSelection}
                        onEnableSelectionMode={() => setIsSelectionMode(true)}
                        onExitSelectionMode={() => setIsSelectionMode(false)}
                        allImages={allImages}
                        onImageClick={(img) => {
                            if (isSelectionMode) toggleImageSelection(img.id);
                            else if (imageUrls[img.id]) setSelectedImage({ image: img, url: imageUrls[img.id] });
                        }}
                        onMenuOpen={handleMenuOpen}
                        gridItemSize={gridItemSize}
                        showNames={showFileNames}
                    />
                )}
            </Box>

            {/* Filter Dialog */}
            <LibraryFilterDialog
                open={filterDialogOpen}
                onClose={() => setFilterDialogOpen(false)}
                dateSections={fullDateSections} // Pass full list for selection
                currentFilter={activeFilter}
                onApply={setActiveFilter}
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
            {/* View Settings Menu */}
            <Menu
                anchorEl={settingsAnchorEl}
                open={Boolean(settingsAnchorEl)}
                onClose={() => setSettingsAnchorEl(null)}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                transformOrigin={{ vertical: 'top', horizontal: 'right' }}
                PaperProps={{
                    elevation: 0,
                    sx: {
                        backgroundColor: theme.palette.mode === 'light' ? 'rgba(255, 255, 255, 0.95)' : 'rgba(45, 45, 45, 0.95)',
                        backdropFilter: 'blur(8px)',
                        borderRadius: '12px',
                        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
                        border: `1px solid ${theme.palette.divider}`,
                        minWidth: '250px',
                        p: 2,
                        mt: 1
                    }
                }}
            >
                <Typography variant="subtitle2" fontWeight="600" sx={{ mb: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    Grid Size
                    <Tooltip title="Reset to Default">
                        <IconButton size="small" onClick={handleResetView}>
                            <ArrowCounterClockwise size={14} />
                        </IconButton>
                    </Tooltip>
                </Typography>
                <Box sx={{ px: 1, mb: 2 }}>
                    <Slider
                        size="small"
                        value={gridItemSize}
                        min={100}
                        max={400}
                        onChange={(_, value) => setGridItemSize(value as number)}
                        valueLabelDisplay="auto"
                        valueLabelFormat={(value) => `${value}px`}
                    />
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                        <Typography variant="caption" color="text.secondary">Small</Typography>
                        <Typography variant="caption" color="text.secondary">Large</Typography>
                    </Box>
                </Box>
                
                <Divider sx={{ my: 1 }} />
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.5 }}>
                    <Typography variant="subtitle2" fontWeight="600">
                        Show File Names
                    </Typography>
                    <Switch 
                        size="small"
                        checked={showFileNames} 
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setShowFileNames(e.target.checked)} 
                    />
                </Box>
            </Menu>

            {/* Selection Bar (Bottom) */}
            {isSelectionMode && (
                <LibrarySelectionBar 
                    selectedCount={selectedImageIds.size}
                    onClose={() => {
                        setIsSelectionMode(false);
                        clearSelection();
                    }}
                    onDelete={handleBatchDelete}
                    onDetect={handleBatchDetect}
                    onSave={handleBatchSave}
                    leftSidebarOpen={leftSidebarOpen}
                    rightSidebarOpen={rightSidebarOpen}
                />
            )}
        </Box>
    );
};

export default LibraryPage;
