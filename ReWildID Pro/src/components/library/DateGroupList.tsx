import { Box, IconButton, Tooltip, Typography, useTheme, Chip, Button } from '@mui/material';
import { AnalyseMenu } from './AnalyseMenu';
import { CaretDown, CaretRight, Check as CheckIcon, DotsThreeVertical, UploadSimple, Images } from '@phosphor-icons/react';
import React, { useEffect, useMemo, useRef, useState, forwardRef, useImperativeHandle, useCallback } from 'react';
import { Virtuoso, VirtuosoHandle } from 'react-virtuoso';
import { DBImage } from '../../types/electron';
import { DateSection, GroupData } from '../../types/library';
import AiModeButton from '../AiModeButton';
import ImageCard from '../ImageCard';

export interface DateGroupListHandle {
    scrollToIndex: (index: number) => void;
    scrollToDate: (date: string) => void;
    scrollToGroup: (groupId: number) => void;
}

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
    showNames?: boolean;
    headerContent?: React.ReactNode;
    onActiveItemChange?: (id: string) => void;
    aspectRatio?: string;
    fullImageUrls?: Record<number, string>;
    loadFullImage?: (image: DBImage) => void;
    // AI Analysis support
    aiButtonMode?: 'detect' | 'reid' | 'analyse';
    onReID?: (images: DBImage[], species: string) => void;
    onClassify?: (images: DBImage[]) => void;
    availableSpecies?: string[];
    // Empty state action
    onUpload?: () => void;
}

type FlatItem =
    | { type: 'date-header'; date: string; id: string }
    | { type: 'group-header'; group: GroupData; id: string }
    | { type: 'image-row'; images: DBImage[]; groupId: number; id: string };

export const DateGroupList = forwardRef<DateGroupListHandle, DateGroupListProps>((props, ref) => {
    const {
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
        showNames = false,
        headerContent,
        onActiveItemChange,
        aspectRatio = '1.618/1',
        fullImageUrls = {},
        loadFullImage,
        aiButtonMode = 'detect',
        onReID,
        onClassify,
        availableSpecies = [],
        onUpload
    } = props;

    const theme = useTheme();
    const [collapsedGroups, setCollapsedGroups] = useState<Set<number>>(new Set());
    const [containerWidth, setContainerWidth] = useState(0);
    const containerRef = useRef<HTMLDivElement>(null);
    const virtuosoRef = useRef<VirtuosoHandle>(null);
    
    // Analyse menu state
    const [analyseMenuOpen, setAnalyseMenuOpen] = useState(false);
    const [analyseMenuGroup, setAnalyseMenuGroup] = useState<GroupData | null>(null);

    // Resize Observer to get width
    useEffect(() => {
        if (!containerRef.current) return;
        const observer = new ResizeObserver(entries => {
            for (const entry of entries) {
                setContainerWidth(entry.contentRect.width);
            }
        });
        observer.observe(containerRef.current);
        return () => observer.disconnect();
    }, []);

    // Calculate columns
    // Account for px: 4 (32px * 2 = 64px) padding on the rows
    const horizontalPadding = 64;
    const availableWidth = containerWidth - horizontalPadding;
    const gap = 16; // 2 * 8px (theme spacing 2)
    const minItemWidth = gridItemSize;

    // Avoid division by zero or negative width
    const columns = availableWidth > 0
        ? Math.max(1, Math.floor((availableWidth + gap) / (minItemWidth + gap)))
        : 0;

    // Flatten Data
    const flatItems = useMemo(() => {
        const items: FlatItem[] = [];
        if (columns === 0) return items;

        dateSections.forEach(section => {
            items.push({ type: 'date-header', date: section.date, id: `date-${section.date}` });

            section.groups.forEach(group => {
                items.push({ type: 'group-header', group, id: `group-${group.id}` });

                if (!collapsedGroups.has(group.id)) {
                    // Chunk images into rows
                    for (let i = 0; i < group.images.length; i += columns) {
                        const rowImages = group.images.slice(i, i + columns);
                        items.push({
                            type: 'image-row',
                            images: rowImages,
                            groupId: group.id,
                            id: `group-${group.id}-row-${i}`
                        });
                    }
                }
            });
        });
        return items;
    }, [dateSections, collapsedGroups, columns]);

    // Expose scrollToIndex to parent via ref
    useImperativeHandle(ref, () => ({
        scrollToIndex: (index: number) => {
            virtuosoRef.current?.scrollToIndex({ index, align: 'start', behavior: 'smooth' });
        },
        scrollToDate: (date: string) => {
            const index = flatItems.findIndex(item => item.type === 'date-header' && item.date === date);
            if (index !== -1) {
                virtuosoRef.current?.scrollToIndex({ index, align: 'start', behavior: 'smooth' });
            }
        },
        scrollToGroup: (groupId: number) => {
            const index = flatItems.findIndex(item => item.type === 'group-header' && item.group.id === groupId);
            if (index !== -1) {
                virtuosoRef.current?.scrollToIndex({ index, align: 'start', behavior: 'smooth' });
            }
        }
    }), [flatItems]);

    // Drag Selection State
    const isPointerDownRef = useRef(false);
    const dragStartIdRef = useRef<number | null>(null);
    const initialSelectionRef = useRef<Set<number>>(new Set());
    const isSelectingRef = useRef(true);
    const autoScrollFrameRef = useRef<number | null>(null);

    // Auto Scroll Logic
    const checkAutoScroll = (clientY: number) => {
        if (!virtuosoRef.current || !isPointerDownRef.current) return;

        const viewportHeight = window.innerHeight;
        const scrollZoneHeight = 100; // px from edge to trigger scroll
        const maxScrollSpeed = 15; // px per frame

        if (autoScrollFrameRef.current) {
            cancelAnimationFrame(autoScrollFrameRef.current);
            autoScrollFrameRef.current = null;
        }

        let scrollAmount = 0;
        if (clientY < scrollZoneHeight) {
            // Scroll Up
            // Speed increases as we get closer to the edge
            const factor = 1 - (clientY / scrollZoneHeight);
            scrollAmount = -Math.max(1, factor * maxScrollSpeed);
        } else if (clientY > viewportHeight - scrollZoneHeight) {
            // Scroll Down
            const factor = 1 - ((viewportHeight - clientY) / scrollZoneHeight);
            scrollAmount = Math.max(1, factor * maxScrollSpeed);
        }

        if (scrollAmount !== 0) {
            virtuosoRef.current.scrollBy({ top: scrollAmount, behavior: 'auto' });
            autoScrollFrameRef.current = requestAnimationFrame(() => checkAutoScroll(clientY));
        }
    };

    // ... (Keeping selection logic helpers but adapted) ...

    const startDragSession = (imgId: number) => {
        dragStartIdRef.current = imgId;
        initialSelectionRef.current = new Set(selectedImageIds);
        const wasSelected = initialSelectionRef.current.has(imgId);
        isSelectingRef.current = !wasSelected;

        const newSelection = new Set(initialSelectionRef.current);
        if (isSelectingRef.current) newSelection.add(imgId);
        else newSelection.delete(imgId);

        if (onSetSelection) onSetSelection(newSelection);
        else onToggleSelection(imgId);
    };

    const handleLongPress = (imgId: number) => {
        if (!isSelectionMode && onEnableSelectionMode) {
            onEnableSelectionMode();
            isPointerDownRef.current = true;
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
            const startIndex = allImages.findIndex(img => img.id === dragStartIdRef.current);
            const currentIndex = allImages.findIndex(img => img.id === imgId);
            if (startIndex === -1 || currentIndex === -1) return;

            const minIndex = Math.min(startIndex, currentIndex);
            const maxIndex = Math.max(startIndex, currentIndex);
            const newSelection = new Set(initialSelectionRef.current);

            for (let i = minIndex; i <= maxIndex; i++) {
                const id = allImages[i].id;
                if (isSelectingRef.current) newSelection.add(id);
                else newSelection.delete(id);
            }
            onSetSelection(newSelection);
        }
    };

    useEffect(() => {
        const handleGlobalPointerMove = (e: PointerEvent) => {
            if (isPointerDownRef.current) {
                checkAutoScroll(e.clientY);
            }
        };

        const handleGlobalPointerUp = () => {
            isPointerDownRef.current = false;
            dragStartIdRef.current = null;
            if (autoScrollFrameRef.current) {
                cancelAnimationFrame(autoScrollFrameRef.current);
                autoScrollFrameRef.current = null;
            }
        };

        window.addEventListener('pointermove', handleGlobalPointerMove);
        window.addEventListener('pointerup', handleGlobalPointerUp);
        return () => {
            window.removeEventListener('pointermove', handleGlobalPointerMove);
            window.removeEventListener('pointerup', handleGlobalPointerUp);
            if (autoScrollFrameRef.current) cancelAnimationFrame(autoScrollFrameRef.current);
        };
    }, []);

    // Actions
    const handleSelectGroup = (groupImages: DBImage[]) => {
        if (!onSetSelection) return;
        if (!isSelectionMode && onEnableSelectionMode) onEnableSelectionMode();

        const groupIds = groupImages.map(img => img.id);
        const allSelected = groupIds.every(id => selectedImageIds.has(id));
        const newSelection = new Set(selectedImageIds);

        if (allSelected) groupIds.forEach(id => newSelection.delete(id));
        else groupIds.forEach(id => newSelection.add(id));

        onSetSelection(newSelection);
        if (newSelection.size === 0 && onExitSelectionMode) onExitSelectionMode();
    };

    const toggleGroup = (groupId: number) => {
        const newCollapsed = new Set(collapsedGroups);
        if (newCollapsed.has(groupId)) newCollapsed.delete(groupId);
        else newCollapsed.add(groupId);
        setCollapsedGroups(newCollapsed);
    };

    const handleDetect = async (images: DBImage[]) => {
        const paths = images.map(img => img.original_path);
        try {
            const response = await window.api.detect(paths, () => { });
            if (!response.ok) alert('Detection failed: ' + response.error);
        } catch (error) {
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

    // Calculate row height for consistent sizing (prevents scroll jumps)
    const getRowHeight = useCallback(() => {
        const [w, h] = aspectRatio.split('/').map(Number);
        return gridItemSize * (h / w) + (showNames ? 40 : 16);
    }, [aspectRatio, gridItemSize, showNames]);

    // Render Item - fixed heights prevent Virtuoso scroll jumps
    const itemContent = (_: number, item: FlatItem) => {
        if (item.type === 'date-header') {
            return (
                <Box id={item.id} sx={{ height: 56, display: 'flex', alignItems: 'flex-end', px: 4, pb: 1 }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, color: theme.palette.text.secondary, textTransform: 'uppercase', letterSpacing: '0.5px', fontSize: '0.875rem' }}>
                        {formatDate(item.date)}
                    </Typography>
                </Box>
            );
        } else if (item.type === 'group-header') {
            const group = item.group;
            const isCollapsed = collapsedGroups.has(group.id);
            const isAllSelected = group.images.every((img: DBImage) => selectedImageIds.has(img.id));

            return (
                <Box id={`group-${group.id}`} sx={{
                    height: 48, // Fixed height prevents scroll jumps
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', px: 4,
                    '&:hover .collapse-arrow': { opacity: 1, transform: 'translateX(0)' },
                    '&:hover .group-menu-button': { opacity: 1 },
                    '&:hover .group-select-button': { opacity: 1 },
                }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, position: 'relative' }}>
                        <Box sx={{ position: 'absolute', left: -29, display: 'flex', alignItems: 'center', height: '100%' }}>
                            <IconButton
                                className="collapse-arrow"
                                size="small"
                                onClick={() => toggleGroup(group.id)}
                                sx={{ padding: 0.5, opacity: 0, transition: 'all 0.2s ease' }}
                            >
                                {isCollapsed ? <CaretRight size={20} /> : <CaretDown size={20} />}
                            </IconButton>
                        </Box>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>{group.name}</Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ bgcolor: theme.palette.action.selected, px: 1, py: 0.5, borderRadius: 1 }}>
                            {group.images.length}
                        </Typography>
                        <Tooltip title="Select all in group">
                            <IconButton
                                className="group-select-button"
                                size="small"
                                onClick={(e) => { e.stopPropagation(); handleSelectGroup(group.images); }}
                                sx={{ opacity: isAllSelected ? 1 : 0, transition: 'opacity 0.2s ease', color: isAllSelected ? 'primary.main' : 'text.secondary' }}
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
                    {aiButtonMode === 'analyse' ? (
                        <>
                            <AiModeButton
                                text={group.images.filter((img: DBImage) => selectedImageIds.has(img.id)).length > 0
                                    ? `Analyse (${group.images.filter((img: DBImage) => selectedImageIds.has(img.id)).length})`
                                    : "Analyse"}
                                onClick={() => {
                                    setAnalyseMenuGroup(group);
                                    setAnalyseMenuOpen(true);
                                }}
                            />
                            {analyseMenuGroup?.id === group.id && analyseMenuGroup && (
                                <AnalyseMenu
                                    open={analyseMenuOpen}
                                    onClose={() => {
                                        setAnalyseMenuOpen(false);
                                        setAnalyseMenuGroup(null);
                                    }}
                                    onClassify={() => {
                                        const groupImages = analyseMenuGroup.images;
                                        const selectedInGroup = groupImages.filter((img: DBImage) => selectedImageIds.has(img.id));
                                        const imagesToProcess = selectedInGroup.length > 0 ? selectedInGroup : groupImages;
                                        if (onClassify) onClassify(imagesToProcess);
                                        setAnalyseMenuOpen(false);
                                        setAnalyseMenuGroup(null);
                                    }}
                                    onReID={(species) => {
                                        const groupImages = analyseMenuGroup.images;
                                        const selectedInGroup = groupImages.filter((img: DBImage) => selectedImageIds.has(img.id));
                                        const imagesToProcess = selectedInGroup.length > 0 ? selectedInGroup : groupImages;
                                        if (onReID) onReID(imagesToProcess, species);
                                        setAnalyseMenuOpen(false);
                                        setAnalyseMenuGroup(null);
                                    }}
                                    availableSpecies={availableSpecies}
                                    selectedCount={
                                        analyseMenuGroup.images.filter((img: DBImage) => selectedImageIds.has(img.id)).length > 0
                                            ? analyseMenuGroup.images.filter((img: DBImage) => selectedImageIds.has(img.id)).length
                                            : analyseMenuGroup.images.length
                                    }
                                />
                            )}
                        </>
                    ) : aiButtonMode === 'reid' ? (
                        <>
                            <AiModeButton
                                text={group.images.filter((img: DBImage) => selectedImageIds.has(img.id)).length > 0
                                    ? `ReID (${group.images.filter((img: DBImage) => selectedImageIds.has(img.id)).length})`
                                    : "ReID"}
                                onClick={() => {
                                    setAnalyseMenuGroup(group);
                                    setAnalyseMenuOpen(true);
                                }}
                            />
                            {analyseMenuGroup?.id === group.id && analyseMenuGroup && (
                                <AnalyseMenu
                                    open={analyseMenuOpen}
                                    onClose={() => {
                                        setAnalyseMenuOpen(false);
                                        setAnalyseMenuGroup(null);
                                    }}
                                    onReID={(species) => {
                                        const groupImages = analyseMenuGroup.images;
                                        const selectedInGroup = groupImages.filter((img: DBImage) => selectedImageIds.has(img.id));
                                        const imagesToProcess = selectedInGroup.length > 0 ? selectedInGroup : groupImages;
                                        if (onReID) onReID(imagesToProcess, species);
                                        setAnalyseMenuOpen(false);
                                        setAnalyseMenuGroup(null);
                                    }}
                                    availableSpecies={availableSpecies}
                                    selectedCount={
                                        analyseMenuGroup.images.filter((img: DBImage) => selectedImageIds.has(img.id)).length > 0
                                            ? analyseMenuGroup.images.filter((img: DBImage) => selectedImageIds.has(img.id)).length
                                            : analyseMenuGroup.images.length
                                    }
                                    reidOnly={true}
                                    title="Re-identification"
                                />
                            )}
                        </>
                    ) : (
                        <AiModeButton
                            text={group.images.filter((img: DBImage) => selectedImageIds.has(img.id)).length > 0
                                ? `Detect (${group.images.filter((img: DBImage) => selectedImageIds.has(img.id)).length})`
                                : "Detect"}
                            onClick={() => {
                                const selectedInGroup = group.images.filter((img: DBImage) => selectedImageIds.has(img.id));
                                handleDetect(selectedInGroup.length > 0 ? selectedInGroup : group.images);
                            }}
                        />
                    )}
                </Box>
            );
        } else {
            // Image Row - fixed height prevents scroll jumps
            const rowHeight = getRowHeight() + 16; // +16 for bottom padding
            return (
                <Box sx={{ height: rowHeight, display: 'grid', gridTemplateColumns: `repeat(${columns}, 1fr)`, gap: 2, pb: 2, px: 4, overflow: 'hidden' }}>
                    {item.images.map(img => {
                        const fileDetails = {
                            name: img.original_path.split(/[\\/]/).pop() || 'image.jpg',
                            path: img.original_path,
                            isDirectory: false
                        };
                        let badge = null;
                        if (img.detections && img.detections.length > 0) {
                            const labels = Array.from(new Set(img.detections.map(d => d.label).filter(l => l && l !== 'blank')));
                            if (labels.length === 0) {
                                badge = <Chip label="Empty" size="small" sx={{ bgcolor: 'rgba(0,0,0,0.6)', color: 'white', height: 20, fontSize: '0.65rem', fontWeight: 600 }} />;
                            } else {
                                const text = labels.length > 1 ? `${labels[0]} +${labels.length - 1}` : labels[0];
                                badge = <Chip label={text} size="small" sx={{ bgcolor: '#ffffff', color: '#000000', height: 20, fontSize: '0.65rem', fontWeight: 600 }} />;
                            }
                        }

                        return (
                            <Box key={img.id} sx={{ minWidth: 0 }}>
                                <ImageCard
                                    file={fileDetails}
                                    date={item.id} // Just needs a string
                                    // @ts-ignore
                                    loadImage={() => {
                                        if (gridItemSize > 500 && loadFullImage) {
                                            loadFullImage(img);
                                        } else {
                                            loadImage(img);
                                        }
                                    }}
                                    imageUrl={(gridItemSize > 500 && fullImageUrls[img.id]) ? fullImageUrls[img.id] : imageUrls[img.id]}
                                    onClick={() => onImageClick(img)}
                                    selectable={isSelectionMode}
                                    selected={selectedImageIds.has(img.id)}
                                    onToggleSelection={() => onToggleSelection(img.id)}
                                    showNames={showNames}
                                    onLongPress={() => handleLongPress(img.id)}
                                    onPointerDown={(e) => handlePointerDown(e, img.id)}
                                    onPointerEnter={() => handlePointerEnter(img.id)}
                                    badge={badge}
                                    aspectRatio={aspectRatio}
                                    isPlaceholder={gridItemSize > 500 && !fullImageUrls[img.id]}
                                />
                            </Box>
                        );
                    })}
                </Box>
            );
        }
    };

    // Memoized components and context for Virtuoso (prevents unnecessary recalculations)
    // NOTE: These hooks MUST be before any early returns to satisfy React's rules of hooks
    const virtuosoComponents = useMemo(() => ({
        Header: () => <Box>{headerContent}</Box>
    }), [headerContent]);

    // Average item height for better Virtuoso estimation
    const avgItemHeight = useMemo(() => {
        const rowHeight = getRowHeight() + 16;
        return Math.round((56 + 48 + rowHeight * 3) / 5);
    }, [getRowHeight]);

    // Empty state - show header content so filter/search controls remain accessible
    if (dateSections.length === 0) {
        return (
            <Box ref={containerRef} sx={{ height: '100%', width: '100%', overflow: 'auto' }}>
                {headerContent}
                <Box sx={{ height: 'calc(100% - 140px)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', opacity: 0.6 }}>
                    <Images size={64} color={theme.palette.text.primary} weight="thin" />
                    <Typography variant="h5" fontWeight="500" sx={{ mt: 3, color: 'text.primary' }}>No images found</Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>Try adjusting your filters or search query</Typography>
                    {onUpload && (
                        <Button
                            variant="contained"
                            startIcon={<UploadSimple size={18} />}
                            onClick={onUpload}
                            sx={{
                                mt: 3,
                                borderRadius: 2,
                                textTransform: 'none',
                                bgcolor: theme.palette.mode === 'dark' ? '#FFFFFF' : '#000000',
                                color: theme.palette.mode === 'dark' ? '#000000' : '#FFFFFF',
                                '&:hover': {
                                    bgcolor: theme.palette.mode === 'dark' ? '#E0E0E0' : '#333333'
                                }
                            }}
                        >
                            Upload
                        </Button>
                    )}
                </Box>
            </Box>
        );
    }

    return (
        <Box ref={containerRef} sx={{ height: '100%', width: '100%' }}>
            <Virtuoso
                ref={virtuosoRef}
                style={{ height: '100%' }}
                data={flatItems}
                itemContent={itemContent}
                overscan={400}
                defaultItemHeight={avgItemHeight}
                computeItemKey={(_, item) => item.id}
                rangeChanged={({ startIndex }) => {
                    if (!onActiveItemChange) return;
                    for (let i = startIndex; i >= 0; i--) {
                        const item = flatItems[i];
                        if (item.type === 'date-header' || item.type === 'group-header') {
                            onActiveItemChange(item.id);
                            break;
                        }
                    }
                }}
                components={virtuosoComponents}
            />
        </Box>
    );
});
