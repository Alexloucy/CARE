import { Box, Collapse, IconButton, Tooltip, Typography, useTheme } from '@mui/material';
import { CaretDown, CaretRight, Check as CheckIcon, DotsThreeVertical, UploadSimple } from '@phosphor-icons/react';
import React, { useEffect, useRef, useState } from 'react';
import { DBImage } from '../../types/electron';
import { DateSection } from '../../types/library';
import AiModeButton from '../AiModeButton';
import ImageCard from '../ImageCard';

interface DateGroupListProps {
    dateSections: DateSection[];
    imageUrls: Record<number, string>;
    loadImage: (image: DBImage) => void;
    isSelectionMode: boolean;
    selectedImageIds: Set<number>;
    onToggleSelection: (id: number) => void;
    onSetSelection?: (ids: Set<number>) => void;
    onEnableSelectionMode?: () => void;
    onExitSelectionMode?: () => void;
    allImages?: DBImage[];
    onImageClick: (image: DBImage) => void;
    onMenuOpen: (event: React.MouseEvent<HTMLElement>, groupId: number) => void;
    gridItemSize?: number;
    showNames?: boolean; // Toggle for showing file names
}

export const DateGroupList: React.FC<DateGroupListProps> = ({
    dateSections,
    imageUrls,
    loadImage,
    isSelectionMode,
    selectedImageIds,
    onToggleSelection,
    onSetSelection,
    onEnableSelectionMode,
    onExitSelectionMode,
    allImages = [],
    onImageClick,
    onMenuOpen,
    gridItemSize = 180,
    showNames = false
}) => {
    const theme = useTheme();
    const [collapsedGroups, setCollapsedGroups] = useState<Set<number>>(new Set());
    
    // Drag Selection State
    const isPointerDownRef = useRef(false);
    const dragStartIdRef = useRef<number | null>(null);
    const initialSelectionRef = useRef<Set<number>>(new Set());
    const isSelectingRef = useRef(true); // true = add to selection, false = remove
    
    // Auto Scroll State
    const scrollIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const scrollContainerRef = useRef<HTMLElement | null>(null);
    const lastPointerYRef = useRef<number>(0);

    const getScrollParent = (node: HTMLElement | null): HTMLElement | null => {
        if (!node) return null;
        const style = window.getComputedStyle(node);
        const overflowY = style.overflowY;
        const isScrollable = overflowY !== 'visible' && overflowY !== 'hidden';
        
        if (isScrollable && node.scrollHeight > node.clientHeight) {
            return node;
        }
        return getScrollParent(node.parentElement);
    };

    const checkAutoScroll = () => {
        if (!isPointerDownRef.current) return;
        
        const y = lastPointerYRef.current;
        const threshold = 100; // px from edge
        const maxSpeed = 20; // px per frame
        
        let speed = 0;
        if (y < threshold) {
            // Scroll Up - faster as we get closer to 0
            speed = -maxSpeed * (1 - y / threshold);
        } else if (y > window.innerHeight - threshold) {
            // Scroll Down
            speed = maxSpeed * (1 - (window.innerHeight - y) / threshold);
        }

        if (speed !== 0) {
            // Find container if needed
            if (!scrollContainerRef.current) {
                // Try to find it starting from a known element ID or just the document body?
                // We can try finding an element we rendered.
                // Let's pick the first date header as a starting point if it exists
                const firstDate = dateSections[0]?.date;
                if (firstDate) {
                    const el = document.getElementById(`date-${firstDate}`);
                    if (el) {
                        scrollContainerRef.current = getScrollParent(el);
                    }
                }
            }

            if (scrollContainerRef.current) {
                scrollContainerRef.current.scrollBy(0, speed);
            } else {
                // Fallback to window
                window.scrollBy(0, speed);
            }
        }
    };

    useEffect(() => {
        const handleGlobalPointerUp = () => {
            isPointerDownRef.current = false;
            dragStartIdRef.current = null;
            if (scrollIntervalRef.current) {
                clearInterval(scrollIntervalRef.current);
                scrollIntervalRef.current = null;
            }
        };

        const handleGlobalPointerMove = (e: PointerEvent) => {
            if (isPointerDownRef.current) {
                lastPointerYRef.current = e.clientY;
                // Ensure loop is running
                if (!scrollIntervalRef.current) {
                    scrollIntervalRef.current = setInterval(checkAutoScroll, 16); // ~60fps
                }
            }
        };

        window.addEventListener('pointerup', handleGlobalPointerUp);
        window.addEventListener('pointermove', handleGlobalPointerMove);
        
        return () => {
            window.removeEventListener('pointerup', handleGlobalPointerUp);
            window.removeEventListener('pointermove', handleGlobalPointerMove);
            if (scrollIntervalRef.current) clearInterval(scrollIntervalRef.current);
        };
    }, [dateSections]); // Re-bind if dateSections change (for getElementById fallback), though unlikely needed.

    const startDragSession = (imgId: number) => {
        dragStartIdRef.current = imgId;
        
        // Snapshot current selection
        initialSelectionRef.current = new Set(selectedImageIds);
        
        // Determine behavior based on the clicked item's state IN THE SNAPSHOT
        // Standard: If clicking unselected -> Select. If clicking selected -> Deselect.
        const wasSelected = initialSelectionRef.current.has(imgId);
        isSelectingRef.current = !wasSelected;

        // Apply to the start item immediately
        const newSelection = new Set(initialSelectionRef.current);
        if (isSelectingRef.current) {
            newSelection.add(imgId);
        } else {
            newSelection.delete(imgId);
        }
        
        if (onSetSelection) {
            onSetSelection(newSelection);
        } else {
            onToggleSelection(imgId);
        }
    };

    const handleLongPress = (imgId: number) => {
        if (!isSelectionMode && onEnableSelectionMode) {
            onEnableSelectionMode();
            
            // Manually start drag session immediately
            isPointerDownRef.current = true;
            
            // Since we are just enabling selection mode, the item is currently unselected (visually).
            // We want to select it and start drag-select mode (adding).
            // Note: selectedImageIds might be stale if onEnableSelectionMode triggers update,
            // but usually it's empty or cleared when entering mode.
            
            // We need to assume 'selectedImageIds' matches what user sees (unselected).
            // But if we reuse startDragSession, it uses current props.
            startDragSession(imgId);
        }
    };

    const handlePointerDown = (_e: React.PointerEvent, imgId: number) => {
        if (isSelectionMode) {
            isPointerDownRef.current = true;
            startDragSession(imgId);
        }
    };

    const handlePointerEnter = (imgId: number) => {
        if (isSelectionMode && isPointerDownRef.current && dragStartIdRef.current !== null && onSetSelection && allImages.length > 0) {
            // Find indices
            // Optimization: We could cache indices map, but findIndex on array of ~thousands is fast enough for UI
            const startIndex = allImages.findIndex(img => img.id === dragStartIdRef.current);
            const currentIndex = allImages.findIndex(img => img.id === imgId);

            if (startIndex === -1 || currentIndex === -1) return;

            const minIndex = Math.min(startIndex, currentIndex);
            const maxIndex = Math.max(startIndex, currentIndex);

            // Start with the INITIAL state (before drag started)
            const newSelection = new Set(initialSelectionRef.current);

            // Apply operation to the range
            for (let i = minIndex; i <= maxIndex; i++) {
                const id = allImages[i].id;
                if (isSelectingRef.current) {
                    newSelection.add(id);
                } else {
                    newSelection.delete(id);
                }
            }

            onSetSelection(newSelection);
        }
    };

    const handleSelectGroup = (groupImages: DBImage[]) => {
        if (!onSetSelection) return;
        
        // Enable selection mode if not active
        if (!isSelectionMode && onEnableSelectionMode) {
            onEnableSelectionMode();
        }

        const groupIds = groupImages.map(img => img.id);
        // Check against current props. Note: If onEnableSelectionMode just ran, selectedImageIds might be empty/stale
        // in this render cycle, so we might default to selecting all.
        const allSelected = groupIds.every(id => selectedImageIds.has(id));
        
        const newSelection = new Set(selectedImageIds);
        
        if (allSelected) {
            // Deselect all
            groupIds.forEach(id => newSelection.delete(id));
        } else {
            // Select all
            groupIds.forEach(id => newSelection.add(id));
        }
        
        onSetSelection(newSelection);

        // Exit selection mode if nothing is left selected
        if (newSelection.size === 0 && onExitSelectionMode) {
            onExitSelectionMode();
        }
    };

    const toggleGroup = (groupId: number) => {
        const newCollapsed = new Set(collapsedGroups);
        if (newCollapsed.has(groupId)) {
            newCollapsed.delete(groupId);
        } else {
            newCollapsed.add(groupId);
        }
        setCollapsedGroups(newCollapsed);
    };

    const handleDetect = async (images: DBImage[]) => {
        console.log('Running detection on', images.length, 'images');
        const paths = images.map(img => img.original_path);
        try {
            const response = await window.api.detect(paths, (text: string) => {
                console.log('Detection Stream:', text);
            });

            if (response.ok) {
                console.log('Detection completed successfully');
                // TODO: Show success notification or navigate
            } else {
                console.error('Detection failed:', response.error);
                alert('Detection failed: ' + response.error);
            }
        } catch (error) {
            console.error('Error triggering detection:', error);
            alert('Error triggering detection: ' + error);
        }
    };

    const formatDate = (dateStr: string) => {
        if (dateStr.length !== 8) return dateStr;
        const year = dateStr.substring(0, 4);
        const month = dateStr.substring(4, 6);
        const day = dateStr.substring(6, 8);
        const date = new Date(parseInt(year), parseInt(month) - 1, parseInt(day));
        return date.toLocaleDateString(undefined, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    };

    if (dateSections.length === 0) {
        return (
            <Box
                sx={{
                    height: '60vh',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    opacity: 0.6
                }}
            >
                <UploadSimple size={64} color={theme.palette.text.primary} weight="thin" />
                <Typography variant="h5" fontWeight="500" sx={{ mt: 3, color: 'text.primary' }}>
                    No images yet
                </Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
                    Drag and drop or click Upload to start
                </Typography>
            </Box>
        );
    }

    return (
        <Box>
            {dateSections.map((section) => (
                <Box key={section.date} id={`date-${section.date}`} sx={{ mb: 5, mt: 2, scrollMarginTop: '100px' }}>
                    <Typography variant="h6" sx={{ mb: 2, fontWeight: 700, color: theme.palette.text.secondary, textTransform: 'uppercase', letterSpacing: '0.5px', fontSize: '0.875rem' }}>
                        {formatDate(section.date)}
                    </Typography>

                    {section.groups.map(group => {
                        const isCollapsed = collapsedGroups.has(group.id);
                        // Check if all in group are selected for button state (optional visual feedback)
                        const isAllSelected = group.images.every(img => selectedImageIds.has(img.id));
                        
                        return (
                            <Box key={group.id} id={`group-${group.id}`} sx={{ mb: 4, scrollMarginTop: '100px' }}>
                                <Box sx={{ 
                                    display: 'flex', 
                                    alignItems: 'center', 
                                    justifyContent: 'space-between', 
                                    mb: 2,
                                    position: 'relative', // Establish positioning context
                                    '&:hover .collapse-arrow': { opacity: 1, transform: 'translateX(0)' },
                                    '&:hover .group-menu-button': { opacity: 1 },
                                    '&:hover .group-select-button': { opacity: 1 },
                                    '& .collapse-arrow': { 
                                        opacity: 0, 
                                        transition: 'all 0.2s ease'
                                    }
                                }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}> {/* Restored gap: 2 */}
                                        {/* Arrow positioned absolutely to the left */}
                                        <Box sx={{ position: 'absolute', left: -29, display: 'flex', alignItems: 'center', height: '100%' }}>
                                            <IconButton
                                                className="collapse-arrow"
                                                size="small"
                                                onClick={() => toggleGroup(group.id)}
                                                sx={{ padding: 0.5 }}
                                            >
                                                {isCollapsed ? <CaretRight size={20} /> : <CaretDown size={20} />}
                                            </IconButton>
                                        </Box>

                                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                            {group.name}
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary" sx={{ bgcolor: theme.palette.action.selected, px: 1, py: 0.5, borderRadius: 1 }}>
                                            {group.images.length}
                                        </Typography>
                                        
                                        {/* Select All Button */}
                                        <Tooltip title="Select all in group" enterDelay={0}>
                                            <IconButton
                                                className="group-select-button"
                                                size="small"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleSelectGroup(group.images);
                                                }}
                                                sx={{ 
                                                    opacity: isAllSelected ? 1 : 0, // Show if selected, else hover
                                                    transition: 'opacity 0.2s ease',
                                                    color: isAllSelected ? 'primary.main' : 'text.secondary'
                                                }}
                                            >
                                                <CheckIcon size={20} weight={isAllSelected ? "bold" : "regular"} />
                                            </IconButton>
                                        </Tooltip>

                                        <IconButton
                                            className="group-menu-button"
                                            size="small"
                                            onClick={(e) => onMenuOpen(e, group.id)}
                                            sx={{ opacity: 0, transition: 'opacity 0.2s ease' }}
                                        >
                                            <DotsThreeVertical size={20} />
                                        </IconButton>
                                    </Box>
                                    <AiModeButton 
                                        text={group.images.filter(img => selectedImageIds.has(img.id)).length > 0 
                                            ? `Detect (${group.images.filter(img => selectedImageIds.has(img.id)).length})` 
                                            : "Detect"}
                                        onClick={() => {
                                            const selectedInGroup = group.images.filter(img => selectedImageIds.has(img.id));
                                            handleDetect(selectedInGroup.length > 0 ? selectedInGroup : group.images);
                                        }} 
                                    />
                                </Box>

                                <Collapse in={!isCollapsed} timeout={300}>
                                    <Box sx={{
                                        display: 'grid',
                                        gridTemplateColumns: `repeat(auto-fill, minmax(${gridItemSize}px, 1fr))`,
                                        gap: 2
                                    }}>
                                        {group.images.map((img) => {
                                            const fileDetails = {
                                                name: img.original_path.split(/[\\/]/).pop() || 'image.jpg',
                                                path: img.original_path,
                                                isDirectory: false
                                            };

                                            return (
                                                <Box key={img.id}>
                                                    <ImageCard
                                                        file={fileDetails}
                                                        date={section.date}
                                                        // @ts-ignore - Wrapper to match prop type if necessary
                                                        loadImage={() => loadImage(img)}
                                                        imageUrl={imageUrls[img.id]}
                                                        onClick={() => onImageClick(img)}
                                                        selectable={isSelectionMode}
                                                        selected={selectedImageIds.has(img.id)}
                                                        onToggleSelection={() => onToggleSelection(img.id)}
                                                        showNames={showNames}
                                                        onLongPress={() => handleLongPress(img.id)}
                                                        onPointerDown={(e) => handlePointerDown(e, img.id)}
                                                        onPointerEnter={() => handlePointerEnter(img.id)}
                                                    />
                                                </Box>
                                            );
                                        })}
                                    </Box>
                                </Collapse>
                            </Box>
                        );
                    })}
                </Box>
            ))}
        </Box>
    );
};
