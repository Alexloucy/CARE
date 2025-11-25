import {
    Box,
    Card,
    Divider,
    GlobalStyles,
    IconButton,
    Menu,
    Skeleton,
    Slider,
    Switch,
    Tooltip,
    Typography,
    useTheme
} from '@mui/material';
import {
    ArrowCounterClockwise,
    CheckSquare,
    Funnel,
    Gear,
    X
} from '@phosphor-icons/react';
import React, { useEffect, useRef, useState } from 'react';
import { NAVBAR_HEIGHT } from '../../app/layout/navbar/Navbar';
import { DBImage } from '../../types/electron';
import { DateSection } from '../../types/library';

// Components
import ImageModal from '../ImageModal';
import { DateGroupList } from './DateGroupList';
import { DragDropOverlay } from './DragDropOverlay';
import { LibraryFilter, LibraryFilterDialog } from './LibraryFilterDialog';
import { LibrarySearchBar } from './LibrarySearchBar';
import { LibrarySelectionBar } from './LibrarySelectionBar';
import { Timeline } from './Timeline';

interface MediaExplorerProps {
    title: string;
    loading: boolean;

    // Data
    dateSections: DateSection[];
    fullDateSections: DateSection[];
    imageUrls: Record<number, string>;
    fullImageUrls: Record<number, string>;
    allImages: DBImage[];
    loadImage: (img: DBImage) => void;
    loadFullImage: (img: DBImage) => void;

    // Filter & Search
    activeFilter: LibraryFilter | null;
    onFilterChange: (filter: LibraryFilter | null) => void;
    searchQuery: string;
    onSearchChange: (query: string) => void;
    filterDialogOpen: boolean;
    setFilterDialogOpen: (open: boolean) => void;

    // Selection
    isSelectionMode: boolean;
    selectedImageIds: Set<number>;
    toggleSelectionMode: () => void;
    toggleImageSelection: (id: number) => void;
    setSelection: (ids: Set<number>) => void;
    clearSelection: () => void;
    setIsSelectionMode: (mode: boolean) => void;

    // Actions
    onBatchDelete: () => void;
    onBatchDetect: () => void;
    onBatchSave: () => void;
    onDeleteImage: (img: DBImage) => Promise<void>; // Single image delete from modal

    // Custom Header Actions (e.g. Upload)
    headerActions?: React.ReactNode;

    // Drag & Drop (Optional)
    onDrop?: (e: React.DragEvent) => void;
    isDragging?: boolean;
    setIsDragging?: (dragging: boolean) => void;

    // Group Menu (Optional)
    onGroupMenuOpen?: (e: React.MouseEvent<HTMLElement>, groupId: number) => void;
    groupMenu?: React.ReactNode;

    // Sidebar State
    leftSidebarOpen: boolean;
    rightSidebarOpen: boolean;

    // Filter Options (for detection page)
    availableSpecies?: string[];
}

export const MediaExplorer: React.FC<MediaExplorerProps> = ({
    title,
    loading,
    dateSections,
    fullDateSections,
    imageUrls,
    fullImageUrls,
    allImages,
    loadImage,
    loadFullImage,
    activeFilter,
    onFilterChange,
    onSearchChange,
    filterDialogOpen,
    setFilterDialogOpen,
    isSelectionMode,
    selectedImageIds,
    toggleSelectionMode,
    toggleImageSelection,
    setSelection,
    clearSelection,
    setIsSelectionMode,
    onBatchDelete,
    onBatchDetect,
    onBatchSave,
    onDeleteImage,
    headerActions,
    onDrop,
    isDragging = false,
    setIsDragging,
    onGroupMenuOpen,
    groupMenu,
    leftSidebarOpen,
    rightSidebarOpen,
    availableSpecies
}) => {
    const theme = useTheme();

    // Local View State
    const [activeId, setActiveId] = useState<string>('');
    const [gridItemSize, setGridItemSize] = useState(180);
    const [showFileNames, setShowFileNames] = useState(false);
    const [settingsAnchorEl, setSettingsAnchorEl] = useState<null | HTMLElement>(null);
    const [selectedImage, setSelectedImage] = useState<{ image: DBImage, url: string } | null>(null);

    // Hotkey: ESC to exit selection mode
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape' && isSelectionMode) {
                clearSelection();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isSelectionMode, clearSelection]);

    // Zoom Handler
    const zoomContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const container = zoomContainerRef.current;
        if (!container) return;

        const handleWheel = (e: WheelEvent) => {
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                const delta = e.deltaY * -2.5;
                setGridItemSize(prev => {
                    const newVal = prev + delta;
                    return Math.min(Math.max(newVal, 100), 400);
                });
            }
        };

        container.addEventListener('wheel', handleWheel, { passive: false });
        return () => container.removeEventListener('wheel', handleWheel);
    }, []);

    const handleResetView = () => {
        setGridItemSize(180);
        setShowFileNames(false);
    };

    // Drag Handlers
    const handleDragOver = (e: React.DragEvent) => {
        if (onDrop && setIsDragging) {
            e.preventDefault();
            setIsDragging(true);
        }
    };
    const handleDragLeave = (e: React.DragEvent) => {
        if (onDrop && setIsDragging) {
            e.preventDefault();
            if (!e.currentTarget.contains(e.relatedTarget as Node)) setIsDragging(false);
        }
    };

    // Scroll Observer
    useEffect(() => {
        if (loading || dateSections.length === 0) return;

        const observer = new IntersectionObserver((entries) => {
            const visibleEntries = entries
                .filter(entry => entry.isIntersecting)
                .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);

            if (visibleEntries.length > 0) {
                setActiveId(visibleEntries[0].target.id);
            }
        }, {
            rootMargin: '0px 0px -80% 0px',
            threshold: 0
        });

        dateSections.forEach(section => {
            section.groups.forEach(group => {
                const groupEl = document.getElementById(`group-${group.id}`);
                if (groupEl) observer.observe(groupEl);
            });
        });

        return () => observer.disconnect();
    }, [loading, dateSections]);

    // Image Modal Logic
    useEffect(() => {
        if (selectedImage) {
            loadFullImage(selectedImage.image);
        }
    }, [selectedImage?.image.id, loadFullImage]);

    useEffect(() => {
        if (selectedImage && fullImageUrls[selectedImage.image.id]) {
            setSelectedImage(prev => prev ? { ...prev, url: fullImageUrls[selectedImage.image.id] } : null);
        }
    }, [fullImageUrls, selectedImage?.image.id]);

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
        document.getElementById(`date-${date}`)?.scrollIntoView({ behavior: 'smooth' });
    };

    const handleGroupClick = (groupId: number) => {
        document.getElementById(`group-${groupId}`)?.scrollIntoView({ behavior: 'smooth' });
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
            onDrop={onDrop}
        >
            <GlobalStyles styles={{
                '*::-webkit-scrollbar': { display: 'none' },
                '*': { scrollbarWidth: 'none', '-ms-overflow-style': 'none' }
            }} />
            <DragDropOverlay isDragging={isDragging} />

            {!loading && dateSections.length > 0 && (
                <Timeline
                    dateSections={dateSections}
                    onDateClick={handleDateClick}
                    onGroupClick={handleGroupClick}
                    activeId={activeId}
                />
            )}

            {/* Content */}
            <Box
                ref={zoomContainerRef}
                sx={{
                    flex: 1,
                    overflow: 'hidden', // Virtualized list handles scrolling
                    p: 0, // Remove padding here, let list items handle it
                    pt: 0,
                }}
            >
                {loading ? (
                    <Box sx={{ p: 4 }}> {/* Add padding back for loading state */}
                        <Skeleton variant="text" sx={{ fontSize: '0.875rem', width: 200, mb: 2 }} />
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                            <Skeleton variant="text" sx={{ fontSize: '1.25rem', width: 150 }} />
                            <Skeleton variant="rounded" width={40} height={24} />
                        </Box>
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
                        onSetSelection={setSelection}
                        onEnableSelectionMode={() => setIsSelectionMode(true)}
                        onExitSelectionMode={() => setIsSelectionMode(false)}
                        allImages={allImages}
                        onImageClick={(img) => {
                            if (isSelectionMode) toggleImageSelection(img.id);
                            else if (imageUrls[img.id]) setSelectedImage({ image: img, url: imageUrls[img.id] });
                        }}
                        onMenuOpen={(e, id) => onGroupMenuOpen && onGroupMenuOpen(e, id)}
                        gridItemSize={gridItemSize}
                        showNames={showFileNames}
                        headerContent={
                            <>
                                <Box sx={{ height: `${NAVBAR_HEIGHT}px` }} />
                                <Box sx={{ p: 3, px: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center', bgcolor: theme.palette.background.default, zIndex: 10 }}>
                                    <Typography variant="h4" fontWeight="bold">{title}</Typography>
                                    <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                                        <LibrarySearchBar onSearch={onSearchChange} />

                                        <Tooltip title="Filter">
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
                                                sx={{ '&:hover': { bgcolor: theme.palette.action.hover } }}
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

                                        {headerActions}
                                    </Box>
                                </Box>
                            </>
                        }
                    />
                )}
            </Box>

            <LibraryFilterDialog
                open={filterDialogOpen}
                onClose={() => setFilterDialogOpen(false)}
                dateSections={fullDateSections}
                currentFilter={activeFilter}
                onApply={onFilterChange}
                availableSpecies={availableSpecies}
            />

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
                onDelete={async () => {
                    if (selectedImage) {
                        await onDeleteImage(selectedImage.image);
                        setSelectedImage(null);
                    }
                }}
                detections={selectedImage?.image.detections}
            />

            {groupMenu}

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

            {isSelectionMode && (
                <LibrarySelectionBar
                    selectedCount={selectedImageIds.size}
                    onClose={() => {
                        setIsSelectionMode(false);
                        clearSelection();
                    }}
                    onDelete={onBatchDelete}
                    onDetect={onBatchDetect}
                    onSave={onBatchSave}
                    leftSidebarOpen={leftSidebarOpen}
                    rightSidebarOpen={rightSidebarOpen}
                />
            )}
        </Box>
    );
};
