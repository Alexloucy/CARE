import { Box, IconButton, Tooltip, Typography, useTheme, Chip } from '@mui/material';
import { CaretDown, CaretRight, Check as CheckIcon, DotsThreeVertical, UploadSimple } from '@phosphor-icons/react';
import React, { useEffect, useMemo, useRef, useState, forwardRef, useImperativeHandle } from 'react';
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
        loadFullImage
    } = props;

    const theme = useTheme();
    const [collapsedGroups, setCollapsedGroups] = useState<Set<number>>(new Set());
    const [containerWidth, setContainerWidth] = useState(0);
    const containerRef = useRef<HTMLDivElement>(null);
    const virtuosoRef = useRef<VirtuosoHandle>(null);

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

    // Auto Scroll State (Simplified or removed if Virtuoso handles scrolling well enough, 
    // but for drag-select we might need custom auto-scroll logic if we want to scroll while dragging at edges.
    // For now, let's keep the logic simple: Standard click/drag selection within view)

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
        const handleGlobalPointerUp = () => {
            isPointerDownRef.current = false;
            dragStartIdRef.current = null;
        };
        window.addEventListener('pointerup', handleGlobalPointerUp);
        return () => window.removeEventListener('pointerup', handleGlobalPointerUp);
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

    // Render Item
    const itemContent = (_: number, item: FlatItem) => {
        if (item.type === 'date-header') {
            return (
                <Box id={item.id} sx={{ mt: 4, mb: 2, px: 4 }}>
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
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2, mt: 1, px: 4,
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
                    <AiModeButton
                        text={group.images.filter((img: DBImage) => selectedImageIds.has(img.id)).length > 0
                            ? `Detect (${group.images.filter((img: DBImage) => selectedImageIds.has(img.id)).length})`
                            : "Detect"}
                        onClick={() => {
                            const selectedInGroup = group.images.filter((img: DBImage) => selectedImageIds.has(img.id));
                            handleDetect(selectedInGroup.length > 0 ? selectedInGroup : group.images);
                        }}
                    />
                </Box>
            );
        } else {
            // Image Row
            return (
                <Box sx={{ display: 'grid', gridTemplateColumns: `repeat(${columns}, 1fr)`, gap: 2, mb: 2, px: 4 }}>
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

    if (dateSections.length === 0) {
        return (
            <Box sx={{ height: '60vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', opacity: 0.6 }}>
                <UploadSimple size={64} color={theme.palette.text.primary} weight="thin" />
                <Typography variant="h5" fontWeight="500" sx={{ mt: 3, color: 'text.primary' }}>No images yet</Typography>
                <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>Drag and drop or click Upload to start</Typography>
            </Box>
        );
    }

    const components = useMemo(() => ({
        Header: ({ context }: { context?: { headerContent: React.ReactNode } }) => (
            <Box>{context?.headerContent}</Box>
        )
    }), []);

    return (
        <Box ref={containerRef} sx={{ height: '100%', width: '100%' }}>
            <Virtuoso
                ref={virtuosoRef}
                style={{ height: '100%' }}
                data={flatItems}
                itemContent={itemContent}
                overscan={500}
                context={{ headerContent }}
                rangeChanged={({ startIndex }) => {
                    if (!onActiveItemChange) return;

                    // Search backward from startIndex to find the most recent header that "owns" this content
                    for (let i = startIndex; i >= 0; i--) {
                        const item = flatItems[i];
                        if (item.type === 'date-header' || item.type === 'group-header') {
                            onActiveItemChange(item.id);
                            break;
                        }
                    }
                }}
                components={components}
            />
        </Box>
    );
});
