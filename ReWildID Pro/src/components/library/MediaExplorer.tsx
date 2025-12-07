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
    ToggleButton,
    ToggleButtonGroup,
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
import { AiModeContext } from '../../contexts/AiModeContext';
import { DBImage } from '../../types/electron';
import { DateSection } from '../../types/library';
import { useColorMode } from '../../features/theme/ThemeContext';

// Components
import ImageModal from '../ImageModal';
import { DateGroupList, DateGroupListHandle } from './DateGroupList';
import { DragDropOverlay } from './DragDropOverlay';
import { LibraryFilter, LibraryFilterDialog } from './LibraryFilterDialog';
import { LibrarySearchBar } from './LibrarySearchBar';
import { LibrarySelectionBar } from './LibrarySelectionBar';
import { Timeline } from './Timeline';

type SortOption = 'default' | 'species' | 'individual' | 'name';

interface MediaExplorerProps {
    title: string;
    loading: boolean;

    // Data
    dateSections: DateSection[];
    fullDateSections: DateSection[];
    sortBy?: SortOption;
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
    onBatchReID?: (species: string) => void;
    onBatchSave: () => void;
    onDeleteImage: (img: DBImage) => Promise<void>; // Single image delete from modal
    onDeleteDetection?: (id: number) => void; // Single detection delete from modal

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

    // AI Analysis support
    aiButtonMode?: 'detect' | 'reid' | 'analyse';

    // ReID Run Filter - only show ReID tags when a specific run is selected
    selectedReidRunId?: number | null;
    onReID?: (images: DBImage[], species: string) => void;
    onClassify?: (images: DBImage[]) => void;

    // Empty state action
    onUpload?: () => void;

    // Scroll state callback
    onScrollStateChange?: (isScrolled: boolean) => void;
}

export const MediaExplorer: React.FC<MediaExplorerProps> = ({
    title,
    loading,
    dateSections,
    fullDateSections,
    sortBy = 'default',
    imageUrls,
    fullImageUrls,
    allImages,
    loadImage,
    loadFullImage,
    activeFilter,
    onFilterChange,
    searchQuery,
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
    onBatchReID,
    onBatchSave,
    onDeleteImage,
    onDeleteDetection,
    headerActions,
    onDrop,
    isDragging = false,
    setIsDragging,
    onGroupMenuOpen,
    groupMenu,
    leftSidebarOpen,
    rightSidebarOpen,
    availableSpecies,
    aiButtonMode = 'detect',
    onReID,
    onClassify,
    onUpload,
    selectedReidRunId = null,
    onScrollStateChange
}) => {
    const theme = useTheme();
    const { colorTheme } = useColorMode();
    const hasGradient = colorTheme.gradient !== 'none' || !!colorTheme.special || !!colorTheme.image;
    // Local View State
    const [activeId, setActiveId] = useState<string>('');
    const [gridItemSize, setGridItemSize] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_gridSize');
        return saved ? parseInt(saved, 10) : 180;
    });
    const [showFileNames, setShowFileNames] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_showNames');
        return saved === 'true';
    });
    const [aspectRatio, setAspectRatio] = useState(() => {
        return localStorage.getItem('mediaExplorer_aspectRatio') || '1.618/1';
    });
    const [useLiquidGlass, setUseLiquidGlass] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_useLiquidGlass');
        return saved === null ? true : saved === 'true';
    });
    const [useRayTracedGlass, setUseRayTracedGlass] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_useRayTracedGlass');
        return saved === null ? true : saved === 'true';
    });
    const [showSpeciesTags, setShowSpeciesTags] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_showSpeciesTags');
        return saved === null ? true : saved === 'true';
    });
    const [showReidTags, setShowReidTags] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_showReidTags');
        return saved === null ? true : saved === 'true';
    });
    const [showBoundingBoxes, setShowBoundingBoxes] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_showBoundingBoxes');
        return saved === null ? true : saved === 'true';
    });

    const [settingsMenuPos, setSettingsMenuPos] = useState<{ top: number; left: number } | null>(null);
    const [selectedImage, setSelectedImage] = useState<{ image: DBImage, url: string } | null>(null);
    const dateGroupListRef = useRef<DateGroupListHandle>(null);

    // Global AI Button Effect State
    const [shouldPlayEffect, setShouldPlayEffect] = useState(false);

    // Trigger effect once on mount
    useEffect(() => {
        setShouldPlayEffect(true);
        // Optional: Auto-off after a timeout as a safety net, though buttons will handle it
        const timer = setTimeout(() => setShouldPlayEffect(false), 3000);
        return () => clearTimeout(timer);
    }, []);

    // Persist Settings
    useEffect(() => {
        localStorage.setItem('mediaExplorer_gridSize', gridItemSize.toString());
        localStorage.setItem('mediaExplorer_showNames', showFileNames.toString());
        localStorage.setItem('mediaExplorer_aspectRatio', aspectRatio);
        localStorage.setItem('mediaExplorer_useLiquidGlass', useLiquidGlass.toString());
        localStorage.setItem('mediaExplorer_useRayTracedGlass', useRayTracedGlass.toString());
        localStorage.setItem('mediaExplorer_showSpeciesTags', showSpeciesTags.toString());
        localStorage.setItem('mediaExplorer_showReidTags', showReidTags.toString());
        localStorage.setItem('mediaExplorer_showBoundingBoxes', showBoundingBoxes.toString());
    }, [gridItemSize, showFileNames, aspectRatio, useLiquidGlass, useRayTracedGlass, showSpeciesTags, showReidTags, showBoundingBoxes]);

    // Sync settings from Settings page (storage event listener)
    useEffect(() => {
        const handleStorageChange = (e: StorageEvent) => {
            if (!e.key?.startsWith('mediaExplorer_')) return;
            const key = e.key.replace('mediaExplorer_', '');
            const value = e.newValue;
            if (value === null) return;

            switch (key) {
                case 'gridSize':
                    setGridItemSize(parseInt(value, 10));
                    break;
                case 'showNames':
                    setShowFileNames(value === 'true');
                    break;
                case 'aspectRatio':
                    setAspectRatio(value);
                    break;
                case 'useLiquidGlass':
                    setUseLiquidGlass(value === 'true');
                    break;
                case 'useRayTracedGlass':
                    setUseRayTracedGlass(value === 'true');
                    break;
                case 'showSpeciesTags':
                    setShowSpeciesTags(value === 'true');
                    break;
                case 'showReidTags':
                    setShowReidTags(value === 'true');
                    break;
                case 'showBoundingBoxes':
                    setShowBoundingBoxes(value === 'true');
                    break;
            }
        };
        window.addEventListener('storage', handleStorageChange);
        return () => window.removeEventListener('storage', handleStorageChange);
    }, []);

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
                    return Math.min(Math.max(newVal, 100), 715);
                });
            }
        };

        container.addEventListener('wheel', handleWheel, { passive: false });
        return () => container.removeEventListener('wheel', handleWheel);
    }, []);

    const handleResetView = () => {
        setGridItemSize(180);
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

    // Active item detection is now handled by DateGroupList's rangeChanged callback

    // Image Modal Logic - Sliding window preloading
    const PRELOAD_WINDOW = 3; // Preload 3 images before and after current

    useEffect(() => {
        if (selectedImage) {
            // Load current image
            loadFullImage(selectedImage.image);

            // Sliding window preload: load nearby images
            const currentIndex = allImages.findIndex(img => img.id === selectedImage.image.id);
            if (currentIndex !== -1) {
                // Preload images in window around current
                for (let offset = 1; offset <= PRELOAD_WINDOW; offset++) {
                    // Preload next images
                    if (currentIndex + offset < allImages.length) {
                        const nextImg = allImages[currentIndex + offset];
                        if (!fullImageUrls[nextImg.id]) {
                            loadFullImage(nextImg);
                        }
                    }
                    // Preload previous images
                    if (currentIndex - offset >= 0) {
                        const prevImg = allImages[currentIndex - offset];
                        if (!fullImageUrls[prevImg.id]) {
                            loadFullImage(prevImg);
                        }
                    }
                }
            }
        }
    }, [selectedImage?.image.id, loadFullImage, allImages, fullImageUrls]);

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
        dateGroupListRef.current?.scrollToDate(date);
    };

    const handleGroupClick = (groupId: number) => {
        dateGroupListRef.current?.scrollToGroup(groupId);
    };

    return (
        <AiModeContext.Provider value={{ shouldPlayEffect, setShouldPlayEffect }}>
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
                        rightSidebarOpen={rightSidebarOpen}
                    />
                )}

                {/* Content */}
                <Box
                    ref={zoomContainerRef}
                    data-tour="library-grid"
                    sx={{
                        flex: 1,
                        overflow: 'hidden', // Virtualized list handles scrolling
                        p: 0, // Remove padding here, let list items handle it
                        pt: 0,
                    }}
                >
                    {loading ? (
                        <Box sx={{ height: '100%', overflow: 'hidden' }}>
                            {/* Header skeleton */}
                            <Box sx={{ height: `${NAVBAR_HEIGHT}px` }} />
                            <Box sx={{ p: 3, px: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Skeleton variant="text" sx={{ fontSize: '2rem', width: 180 }} />
                                <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                                    <Skeleton variant="rounded" width={36} height={36} sx={{ borderRadius: 2 }} />
                                    <Skeleton variant="circular" width={36} height={36} />
                                    <Skeleton variant="circular" width={36} height={36} />
                                    <Skeleton variant="circular" width={36} height={36} />
                                </Box>
                            </Box>

                            {/* Content skeleton */}
                            <Box sx={{ px: 4, pt: 2 }}>
                                {/* Date header skeleton */}
                                <Skeleton variant="text" sx={{ fontSize: '0.875rem', width: 140, mb: 2 }} />

                                {/* Group header skeleton */}
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                                    <Skeleton variant="text" sx={{ fontSize: '1.25rem', width: 200 }} />
                                    <Skeleton variant="rounded" width={40} height={24} sx={{ borderRadius: 1 }} />
                                </Box>

                                {/* Grid skeleton */}
                                <Box sx={{ display: 'grid', gridTemplateColumns: `repeat(auto-fill, minmax(${gridItemSize}px, 1fr))`, gap: 2 }}>
                                    {[...Array(12)].map((_, i) => (
                                        <Card key={i} sx={{ aspectRatio: aspectRatio, width: '100%', borderRadius: 3, boxShadow: 'none', overflow: 'hidden' }}>
                                            <Skeleton variant="rectangular" width="100%" height="100%" animation="wave" />
                                        </Card>
                                    ))}
                                </Box>
                            </Box>
                        </Box>
                    ) : (
                        <DateGroupList
                            ref={dateGroupListRef}
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
                            aspectRatio={aspectRatio}
                            fullImageUrls={fullImageUrls}
                            loadFullImage={loadFullImage}
                            onActiveItemChange={setActiveId}
                            aiButtonMode={aiButtonMode}
                            onReID={onReID}
                            onClassify={onClassify}
                            availableSpecies={availableSpecies}
                            onUpload={onUpload}
                            sortBy={sortBy}
                            showSpeciesTags={showSpeciesTags}
                            showReidTags={showReidTags && selectedReidRunId !== null}
                            onScrollStateChange={onScrollStateChange}
                            headerContent={
                                <>
                                    <Box sx={{ height: `${NAVBAR_HEIGHT}px` }} />
                                    <Box sx={{
                                        p: 3,
                                        px: 4,
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        bgcolor: hasGradient
                                            ? "transparent"
                                            : theme.palette.background.default,
                                        zIndex: 10
                                    }}>
                                        <Typography variant="h4" fontWeight="bold">{title}</Typography>
                                        <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                                            <LibrarySearchBar value={searchQuery} onSearch={onSearchChange} />

                                            <Tooltip title="Filter">
                                                <IconButton
                                                    data-tour="library-filter"
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
                                                    onClick={(e: React.MouseEvent<HTMLButtonElement>) => {
                                                        const rect = e.currentTarget.getBoundingClientRect();
                                                        setSettingsMenuPos({ top: rect.bottom, left: rect.right });
                                                    }}
                                                    sx={{ '&:hover': { bgcolor: theme.palette.action.hover } }}
                                                >
                                                    <Gear weight="regular" />
                                                </IconButton>
                                            </Tooltip>

                                            <Tooltip title={isSelectionMode ? "Cancel Selection" : "Select Items"}>
                                                <IconButton
                                                    data-tour="library-select"
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

                {/* Settings Menu - rendered at root level to avoid re-renders */}
                <Menu
                    open={Boolean(settingsMenuPos)}
                    onClose={() => setSettingsMenuPos(null)}
                    anchorReference="anchorPosition"
                    anchorPosition={settingsMenuPos ? { top: settingsMenuPos.top, left: settingsMenuPos.left } : undefined}
                    transformOrigin={{ vertical: 'top', horizontal: 'right' }}
                    slotProps={{
                        paper: {
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
                            max={715}
                            onChange={(_: Event, value: number | number[]) => setGridItemSize(value as number)}
                            valueLabelDisplay="auto"
                            valueLabelFormat={(value: number) => `${value}px`}
                        />
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                            <Typography variant="caption" color="text.secondary">Small</Typography>
                            <Typography variant="caption" color="text.secondary">Large</Typography>
                        </Box>
                    </Box>

                    <Divider sx={{ my: 1 }} />

                    <Typography variant="subtitle2" fontWeight="600" sx={{ mb: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        Aspect Ratio
                        <Tooltip title="Reset to Default">
                            <IconButton size="small" onClick={() => setAspectRatio('1.618/1')}>
                                <ArrowCounterClockwise size={14} />
                            </IconButton>
                        </Tooltip>
                    </Typography>
                    <ToggleButtonGroup
                        value={aspectRatio}
                        exclusive
                        onChange={(_: React.MouseEvent<HTMLElement>, value: string | null) => value && setAspectRatio(value)}
                        size="small"
                        fullWidth
                        sx={{ mb: 2, display: 'flex' }}
                    >
                        <ToggleButton value="1/1" sx={{ flexGrow: 1, py: 0.5 }}>1:1</ToggleButton>
                        <ToggleButton value="4/3" sx={{ flexGrow: 1, py: 0.5 }}>4:3</ToggleButton>
                        <ToggleButton value="16/9" sx={{ flexGrow: 1, py: 0.5 }}>16:9</ToggleButton>
                        <ToggleButton value="9/16" sx={{ flexGrow: 1, py: 0.5 }}>9:16</ToggleButton>
                        <ToggleButton value="1.618/1" sx={{ flexGrow: 1, py: 0.5 }}>Φ</ToggleButton>
                        <ToggleButton value="1/1.618" sx={{ flexGrow: 1, py: 0.5 }}>Φ</ToggleButton>
                    </ToggleButtonGroup>

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
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.5 }}>
                        <Typography variant="subtitle2" fontWeight="600">
                            Liquid Glass BBox
                        </Typography>
                        <Switch
                            size="small"
                            checked={useLiquidGlass}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUseLiquidGlass(e.target.checked)}
                        />
                    </Box>
                    {useLiquidGlass && (
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.5, pl: 2 }}>
                            <Typography variant="caption" color="text.secondary">
                                Ray-traced Glass
                            </Typography>
                            <Switch
                                size="small"
                                checked={useRayTracedGlass}
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUseRayTracedGlass(e.target.checked)}
                            />
                        </Box>
                    )}

                    <Divider sx={{ my: 1 }} />

                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.5 }}>
                        <Typography variant="subtitle2" fontWeight="600">
                            Show Species Tags
                        </Typography>
                        <Switch
                            size="small"
                            checked={showSpeciesTags}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setShowSpeciesTags(e.target.checked)}
                        />
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.5 }}>
                        <Typography variant="subtitle2" fontWeight="600">
                            Show Individual Tags
                        </Typography>
                        <Switch
                            size="small"
                            checked={showReidTags}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setShowReidTags(e.target.checked)}
                        />
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.5 }}>
                        <Typography variant="subtitle2" fontWeight="600">
                            Show Bounding Boxes
                        </Typography>
                        <Switch
                            size="small"
                            checked={showBoundingBoxes}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setShowBoundingBoxes(e.target.checked)}
                        />
                    </Box>
                </Menu>

                <ImageModal
                    open={!!selectedImage}
                    onClose={() => setSelectedImage(null)}
                    imageUrl={selectedImage?.url}
                    imageId={selectedImage?.image.id}
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
                    detections={showBoundingBoxes ? selectedImage?.image.detections : []}
                    reidResults={selectedImage?.image.reidResults}
                    useLiquidGlass={useLiquidGlass}
                    useRayTracedGlass={useRayTracedGlass}
                    onDeleteDetection={onDeleteDetection}
                />

                {groupMenu}

                {isSelectionMode && (
                    <LibrarySelectionBar
                        selectedCount={selectedImageIds.size}
                        onClose={() => {
                            setIsSelectionMode(false);
                            clearSelection();
                        }}
                        onDelete={onBatchDelete}
                        onClassify={onBatchDetect}
                        onReID={onBatchReID || (() => { })}
                        onSave={onBatchSave}
                        leftSidebarOpen={leftSidebarOpen}
                        rightSidebarOpen={rightSidebarOpen}
                        availableSpecies={availableSpecies}
                    />
                )}
            </Box>
        </AiModeContext.Provider>
    );
};
