import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useOutletContext, useNavigate } from 'react-router-dom';
import {
    Box, Typography, IconButton, Menu, MenuItem,
    Chip, alpha, useTheme, Collapse, Skeleton, Button,
    Switch, Tooltip, Slider
} from '@mui/material';
import { Virtuoso } from 'react-virtuoso';
import {
    ArrowLineUp, Fingerprint, DotsThreeVertical, PencilSimple, Trash, CaretDown, CaretRight,
    Images as ImagesIcon, CaretLeft, Sparkle, Gear, ArrowCounterClockwise
} from '@phosphor-icons/react';
import { LiquidGlassButton } from '../../components/LiquidGlassButton';
import { GroupNameDialog } from '../../components/GroupNameDialog';
import { LibrarySearchBar } from '../../components/library/LibrarySearchBar';
import ImageModal from '../../components/ImageModal';
import { RefreshNotification } from '../../components/RefreshNotification';
import { Detection, DBImage } from '../../types/electron';

interface ReidRun { id: number; name: string; species: string; created_at: number; individual_count: number; detection_count: number; }
interface ReidDetection { id: number; image_id: number; label: string; confidence: number; detection_confidence: number; x1: number; y1: number; x2: number; y2: number; source: string; batch_id: number; created_at: number; image_path: string; image_preview_path?: string; }
interface ReidIndividual { id: number; run_id: number; name: string; display_name: string; color: string; created_at: number; member_count: number; detections: ReidDetection[]; }

// Skeleton Card for loading state
const SkeletonCard: React.FC = () => {
    const theme = useTheme();
    return (
        <Box sx={{ borderRadius: 2, overflow: 'hidden', border: `1px solid ${theme.palette.divider}`, bgcolor: theme.palette.mode === 'light' ? '#F7F9FB' : theme.palette.background.paper }}>
            <Skeleton variant="rectangular" width="100%" height={130} animation="wave" />
            <Box sx={{ p: 1.25 }}>
                <Skeleton variant="text" width="70%" height={20} animation="wave" />
                <Skeleton variant="text" width="50%" height={16} animation="wave" />
            </Box>
        </Box>
    );
};

const IndividualCard: React.FC<{ individual: ReidIndividual; onClick: () => void; imageUrls: Map<string, string> }> = ({ individual, onClick, imageUrls }) => {
    const theme = useTheme();
    const firstDet = individual.detections[0];
    const thumbUrl = firstDet ? imageUrls.get(firstDet.image_preview_path || firstDet.image_path) : undefined;
    return (
        <Box onClick={onClick} sx={{ cursor: 'pointer', borderRadius: 2, overflow: 'hidden', transition: 'all 0.15s', border: `1px solid ${theme.palette.divider}`, bgcolor: theme.palette.mode === 'light' ? '#F7F9FB' : theme.palette.background.paper, '&:hover': { borderColor: individual.color } }}>
            <Box sx={{ width: '100%', height: 130, bgcolor: theme.palette.mode === 'light' ? '#f0f0f0' : '#0a0a0a', position: 'relative', overflow: 'hidden' }}>
                {thumbUrl ? <Box component="img" src={thumbUrl} sx={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : <Box sx={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Fingerprint size={40} weight="thin" color={theme.palette.text.disabled} /></Box>}
                <Box sx={{ position: 'absolute', top: 8, left: 8, width: 12, height: 12, borderRadius: '50%', bgcolor: individual.color, border: '2px solid white', boxShadow: '0 1px 3px rgba(0,0,0,0.3)' }} />
                <Box sx={{ position: 'absolute', bottom: 6, right: 6, display: 'flex', alignItems: 'center', gap: 0.4, bgcolor: 'rgba(0,0,0,0.6)', color: 'white', px: 0.8, py: 0.3, borderRadius: 1, fontSize: '12px' }}>
                    <ImagesIcon size={14} />{individual.member_count}
                </Box>
            </Box>
            <Box sx={{ p: 1.25 }}>
                <Typography variant="body2" fontWeight={600} noWrap sx={{ color: individual.color }}>{individual.display_name}</Typography>
                <Typography variant="caption" color="text.secondary">{individual.member_count} sighting{individual.member_count !== 1 ? 's' : ''}</Typography>
            </Box>
        </Box>
    );
};

interface RunGroupProps {
    run: ReidRun;
    individuals: ReidIndividual[];
    imageUrls: Map<string, string>;
    onIndividualClick: (ind: ReidIndividual) => void;
    onMenuOpen: (e: React.MouseEvent<HTMLElement>, runId: number) => void;
    hasMore: boolean;
    loadingMore: boolean;
    onLoadMore: () => void;
}

const RunGroup: React.FC<RunGroupProps> = ({ run, individuals, imageUrls, onIndividualClick, onMenuOpen, hasMore, loadingMore, onLoadMore }) => {
    const theme = useTheme();
    const [expanded, setExpanded] = useState(true);
    const loadMoreRef = useRef<HTMLDivElement>(null);
    const formatDate = (ts: number) => new Date(ts).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });

    // Intersection observer for infinite scroll
    useEffect(() => {
        if (!expanded || !hasMore || loadingMore) return;
        
        const observer = new IntersectionObserver(
            (entries) => {
                if (entries[0].isIntersecting && hasMore && !loadingMore) {
                    onLoadMore();
                }
            },
            { threshold: 0.1 }
        );
        
        if (loadMoreRef.current) observer.observe(loadMoreRef.current);
        return () => observer.disconnect();
    }, [expanded, hasMore, loadingMore, onLoadMore]);

    return (
        <Box sx={{ mb: 3 }}>
            <Box onClick={() => setExpanded(!expanded)} sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1.5, borderRadius: 2, bgcolor: alpha(theme.palette.text.primary, 0.03), mb: expanded ? 2 : 0, cursor: 'pointer', '&:hover': { bgcolor: alpha(theme.palette.text.primary, 0.05) } }}>
                <IconButton size="small" sx={{ p: 0.5 }}>{expanded ? <CaretDown size={18} /> : <CaretRight size={18} />}</IconButton>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1 }}>
                    <Fingerprint size={20} weight="duotone" />
                    <Typography fontWeight={600}>{run.name}</Typography>
                    <Chip size="small" label={run.species} sx={{ height: 22, fontSize: '0.75rem' }} />
                </Box>
                <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>{run.individual_count} individual{run.individual_count !== 1 ? 's' : ''} • {run.detection_count} detection{run.detection_count !== 1 ? 's' : ''} • {formatDate(run.created_at)}</Typography>
                <IconButton size="small" onClick={(e: React.MouseEvent<HTMLButtonElement>) => { e.stopPropagation(); onMenuOpen(e, run.id); }}><DotsThreeVertical size={18} /></IconButton>
            </Box>
            <Collapse in={expanded}>
                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 2, pl: 4 }}>
                    {individuals.map((ind) => <IndividualCard key={ind.id} individual={ind} onClick={() => onIndividualClick(ind)} imageUrls={imageUrls} />)}
                    {loadingMore && Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={`skeleton-${i}`} />)}
                </Box>
                {hasMore && <Box ref={loadMoreRef} sx={{ height: 20, mt: 2 }} />}
            </Collapse>
        </Box>
    );
};

// Detail view for viewing an individual's images - copies DateGroupList structure exactly
interface IndividualDetailViewProps {
    individual: ReidIndividual;
    onBack: () => void;
}

const IndividualDetailView: React.FC<IndividualDetailViewProps> = ({ 
    individual, 
    onBack
}) => {
    const theme = useTheme();
    const [loading, setLoading] = useState(true);
    const [dbImages, setDbImages] = useState<DBImage[]>([]);
    const [imageUrls, setImageUrls] = useState<Record<number, string>>({});
    const [fullImageUrls, setFullImageUrls] = useState<Record<number, string>>({});
    const [selectedImage, setSelectedImage] = useState<DBImage | null>(null);
    const [gridItemSize, setGridItemSize] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_gridSize');
        return saved ? parseInt(saved, 10) : 180;
    });
    const [useLiquidGlass] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_useLiquidGlass');
        return saved === null ? true : saved === 'true';
    });
    const [useRayTracedGlass] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_useRayTracedGlass');
        return saved === null ? true : saved === 'true';
    });
    const [containerWidth, setContainerWidth] = useState(0);
    const containerRef = useRef<HTMLDivElement>(null);
    const [settingsMenuPos, setSettingsMenuPos] = useState<{ top: number; left: number } | null>(null);

    // Zoom Handler - Callback ref ensures listener is attached immediately when node exists
    const zoomRef = useCallback((node: HTMLDivElement | null) => {
        if (!node) return;

        const handleWheel = (e: WheelEvent) => {
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                const delta = e.deltaY * -2.5;
                setGridItemSize(prev => Math.min(Math.max(prev + delta, 100), 715));
            }
        };

        node.addEventListener('wheel', handleWheel, { passive: false });
        
        // Cleanup listener when node changes or unmounts
        return () => {
            node.removeEventListener('wheel', handleWheel);
        };
    }, []);

    // Fetch DBImage objects for this individual's detections
    useEffect(() => {
        const fetchImages = async () => {
            setLoading(true);
            const imageIds = [...new Set(individual.detections.map(d => d.image_id))];
            if (imageIds.length > 0) {
                try {
                    const result = await window.api.getImagesByIds(imageIds);
                    if (result.ok && result.images) {
                        setDbImages(result.images);
                    }
                } catch (error) {
                    console.error('[IndividualDetailView] Error fetching images:', error);
                }
            }
            setLoading(false);
        };
        fetchImages();
    }, [individual.id]);

    // Measure container width (same as DateGroupList)
    // Re-run when loading changes because containerRef is only available after loading
    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;
        const observer = new ResizeObserver(entries => {
            for (const entry of entries) {
                setContainerWidth(entry.contentRect.width);
            }
        });
        observer.observe(container);
        return () => observer.disconnect();
    }, [loading]);

    // Load thumbnails
    useEffect(() => {
        const loadThumbnails = async () => {
            for (const img of dbImages) {
                if (!imageUrls[img.id]) {
                    const path = img.preview_path || img.original_path;
                    try {
                        const response = await window.api.viewImage(path);
                        if (response.ok && response.data) {
                            const blob = new Blob([response.data as unknown as BlobPart], { type: 'image/jpeg' });
                            const url = URL.createObjectURL(blob);
                            setImageUrls(prev => ({ ...prev, [img.id]: url }));
                        }
                    } catch (e) { /* ignore */ }
                }
            }
        };
        if (dbImages.length > 0) loadThumbnails();
    }, [dbImages]);

    // Load full image for modal
    const loadFullImage = useCallback(async (img: DBImage) => {
        if (!fullImageUrls[img.id]) {
            try {
                const response = await window.api.viewImage(img.original_path);
                if (response.ok && response.data) {
                    const blob = new Blob([response.data as unknown as BlobPart], { type: 'image/jpeg' });
                    const url = URL.createObjectURL(blob);
                    setFullImageUrls(prev => ({ ...prev, [img.id]: url }));
                }
            } catch (e) { /* ignore */ }
        }
    }, [fullImageUrls]);


    // Column calculation (copied exactly from DateGroupList)
    const horizontalPadding = 64; // px: 4 = 32px * 2
    const gap = 16; // theme spacing 2
    const availableWidth = containerWidth - horizontalPadding;
    const columns = availableWidth > 0 ? Math.max(1, Math.floor((availableWidth + gap) / (gridItemSize + gap))) : 1;
    const actualItemWidth = columns > 0 ? (availableWidth - (gap * (columns - 1))) / columns : gridItemSize;

    // Row height calculation (copied from DateGroupList)
    const aspectRatio = '1.618/1';
    const getRowHeight = useCallback(() => {
        const [w, h] = aspectRatio.split('/').map(Number);
        return actualItemWidth * (h / w) + 16;
    }, [actualItemWidth]);

    // Flatten images into rows
    const imageRows = useMemo(() => {
        if (columns === 0) return [];
        const rows: DBImage[][] = [];
        for (let i = 0; i < dbImages.length; i += columns) {
            rows.push(dbImages.slice(i, i + columns));
        }
        return rows;
    }, [dbImages, columns]);

    // Get detection for current image
    const getDetectionsForImage = useCallback((imgId: number): Detection[] => {
        return individual.detections
            .filter(d => d.image_id === imgId)
            .map(d => ({
                id: d.id,
                image_id: d.image_id,
                label: individual.display_name,
                confidence: d.confidence,
                detection_confidence: d.detection_confidence,
                x1: d.x1,
                y1: d.y1,
                x2: d.x2,
                y2: d.y2,
                source: d.source,
                batch_id: d.batch_id,
                created_at: d.created_at
            } as Detection));
    }, [individual]);

    // Navigation in modal
    const currentIndex = selectedImage ? dbImages.findIndex(img => img.id === selectedImage.id) : -1;
    const hasNext = currentIndex >= 0 && currentIndex < dbImages.length - 1;
    const hasPrev = currentIndex > 0;
    const goNext = useCallback(() => { 
        if (hasNext) { 
            const nextImg = dbImages[currentIndex + 1]; 
            setSelectedImage(nextImg); 
            loadFullImage(nextImg); 
        } 
    }, [hasNext, currentIndex, dbImages, loadFullImage]);
    const goPrev = useCallback(() => { 
        if (hasPrev) { 
            const prevImg = dbImages[currentIndex - 1]; 
            setSelectedImage(prevImg); 
            loadFullImage(prevImg); 
        } 
    }, [hasPrev, currentIndex, dbImages, loadFullImage]);

    // Handle image click
    const handleImageClick = useCallback((img: DBImage) => {
        setSelectedImage(img);
        loadFullImage(img);
    }, [loadFullImage]);

    // Preload nearby images when modal is open
    useEffect(() => {
        if (selectedImage && currentIndex !== -1) {
            for (let offset = 1; offset <= 3; offset++) {
                if (currentIndex + offset < dbImages.length) loadFullImage(dbImages[currentIndex + offset]);
                if (currentIndex - offset >= 0) loadFullImage(dbImages[currentIndex - offset]);
            }
        }
    }, [selectedImage, currentIndex, dbImages, loadFullImage]);

    // Header component that scrolls with content (via Virtuoso)
    // Includes navbar spacer (64px) like MediaExplorer
    const headerContent = useMemo(() => (
        <Box>
            {/* Navbar spacer */}
            <Box sx={{ height: 64 }} />
            {/* Individual header */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 2, px: 4 }}>
                <IconButton onClick={onBack} sx={{ bgcolor: alpha(theme.palette.text.primary, 0.05), '&:hover': { bgcolor: alpha(theme.palette.text.primary, 0.1) } }}>
                    <CaretLeft size={20} />
                </IconButton>
                <Box sx={{ width: 20, height: 20, borderRadius: '50%', bgcolor: individual.color, border: '2px solid', borderColor: theme.palette.background.paper, boxShadow: 1 }} />
                <Box sx={{ flex: 1 }}>
                    <Typography variant="h5" fontWeight={600}>{individual.display_name}</Typography>
                    <Typography variant="body2" color="text.secondary">
                        {individual.member_count} sighting{individual.member_count !== 1 ? 's' : ''} • {dbImages.length} image{dbImages.length !== 1 ? 's' : ''}
                    </Typography>
                </Box>
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
            </Box>
        </Box>
    ), [individual, dbImages.length, onBack, theme, setSettingsMenuPos]);

    // Virtuoso components with Header (scrolls with content)
    const virtuosoComponents = useMemo(() => ({
        Header: () => headerContent
    }), [headerContent]);

    if (loading) {
        return (
            <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', pt: '64px' }}>
                <Typography color="text.secondary">Loading images...</Typography>
            </Box>
        );
    }

    const rowHeight = getRowHeight() + 24; // +24 for bottom margin (pb: 3)

    return (
        <>
        <Box
            ref={zoomRef}
            sx={{
                flex: 1,
                overflow: 'hidden',
                height: '100%',
                width: '100%'
            }}
        >
            <Box ref={containerRef} sx={{ height: '100%', width: '100%' }}>
                <Virtuoso
                style={{ height: '100%' }}
                totalCount={imageRows.length}
                defaultItemHeight={rowHeight}
                components={virtuosoComponents}
                itemContent={(rowIndex: number) => {
                    const row = imageRows[rowIndex];
                    return (
                        <Box sx={{ 
                            height: rowHeight,
                            display: 'grid', 
                            gridTemplateColumns: `repeat(${columns}, 1fr)`,
                            gap: 2,
                            pb: 3,
                            px: 4,
                            overflow: 'hidden',
                            alignItems: 'start'
                        }}>
                            {row.map(img => {
                                const url = imageUrls[img.id];
                                return (
                                    <Box
                                        key={img.id}
                                        onClick={() => handleImageClick(img)}
                                        sx={{
                                            minWidth: 0,
                                            height: 'fit-content',
                                            aspectRatio,
                                            borderRadius: 1.5,
                                            overflow: 'hidden',
                                            cursor: 'pointer',
                                            bgcolor: theme.palette.mode === 'light' ? '#f5f5f5' : '#1a1a1a',
                                            transition: 'transform 0.15s, box-shadow 0.15s',
                                            '&:hover': {
                                                transform: 'scale(1.02)',
                                                boxShadow: 3
                                            }
                                        }}
                                    >
                                        {url ? (
                                            <Box
                                                component="img"
                                                src={url}
                                                sx={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                            />
                                        ) : (
                                            <Box sx={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                                <Fingerprint size={32} weight="thin" color={theme.palette.text.disabled} />
                                            </Box>
                                        )}
                                    </Box>
                                );
                            })}
                        </Box>
                    );
                }}
            />

            </Box>
        </Box>

        {/* Settings Menu - rendered at root level outside scrollable container */}
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
                    <IconButton size="small" onClick={() => setGridItemSize(180)}>
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
        </Menu>

        {/* Image Modal - rendered at root level outside scrollable container */}
        {(() => {
            const isOpen = selectedImage !== null;
            const imageUrl = selectedImage ? (fullImageUrls[selectedImage.id] || imageUrls[selectedImage.id]) : undefined;
            const dets = selectedImage ? getDetectionsForImage(selectedImage.id) : [];
            const file = selectedImage ? {
                name: selectedImage.original_path.split('\\').pop() || selectedImage.original_path.split('/').pop() || 'unknown',
                isDirectory: false,
                path: selectedImage.original_path
            } : undefined;
            return (
                <ImageModal
                    open={isOpen}
                    onClose={() => setSelectedImage(null)}
                    imageUrl={imageUrl}
                    file={file}
                    detections={dets}
                    onNext={hasNext ? goNext : undefined}
                    onPrev={hasPrev ? goPrev : undefined}
                    hasNext={hasNext}
                    hasPrev={hasPrev}
                    useLiquidGlass={useLiquidGlass}
                    useRayTracedGlass={useRayTracedGlass}
                />
            );
        })()}
    </>
    );
};

const PAGE_SIZE = 12;

const ReIDPage: React.FC = () => {
    const theme = useTheme();
    const navigate = useNavigate();
    useOutletContext<{ leftSidebarOpen: boolean; rightSidebarOpen: boolean }>();
    const [loading, setLoading] = useState(true);
    const [runs, setRuns] = useState<ReidRun[]>([]);
    const [individuals, setIndividuals] = useState<Map<number, ReidIndividual[]>>(new Map()); // runId -> individuals
    const [pagination, setPagination] = useState<Map<number, { page: number; hasMore: boolean }>>(new Map());
    const [loadingMore, setLoadingMore] = useState<Map<number, boolean>>(new Map());
    const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
    const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
    const [renameDialogOpen, setRenameDialogOpen] = useState(false);
    const [runToRename, setRunToRename] = useState<{ id: number; name: string } | null>(null);
    const [selectedIndividual, setSelectedIndividual] = useState<ReidIndividual | null>(null);
    const [imageUrls, setImageUrls] = useState<Map<string, string>>(new Map());
    const [refreshTrigger, setRefreshTrigger] = useState(0);
    
    // Read liquid glass settings from localStorage (shared with MediaExplorer and Settings page)
    const [useLiquidGlass, setUseLiquidGlass] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_useLiquidGlass');
        return saved === null ? true : saved === 'true';
    });
    const [useRayTracedGlass, setUseRayTracedGlass] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_useRayTracedGlass');
        return saved === null ? true : saved === 'true';
    });

    // Sync settings from Settings page
    useEffect(() => {
        const handleStorageChange = (e: StorageEvent) => {
            if (e.key === 'mediaExplorer_useLiquidGlass' && e.newValue !== null) {
                setUseLiquidGlass(e.newValue === 'true');
            } else if (e.key === 'mediaExplorer_useRayTracedGlass' && e.newValue !== null) {
                setUseRayTracedGlass(e.newValue === 'true');
            }
        };
        window.addEventListener('storage', handleStorageChange);
        return () => window.removeEventListener('storage', handleStorageChange);
    }, []);
    
    // Search and settings menu
    const [searchQuery, setSearchQuery] = useState('');
    const [settingsMenuPos, setSettingsMenuPos] = useState<{ top: number; left: number } | null>(null);

    // Scroll state for floating button
    const [isScrolled, setIsScrolled] = useState(false);

    const refreshData = useCallback(() => {
        setRefreshTrigger(prev => prev + 1);
    }, []);

    // Scroll detection
    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 150);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    // Listen for refresh events from TaskPanel
    useEffect(() => {
        const handleRefresh = (e: CustomEvent<{ page: string }>) => {
            if (e.detail.page === 'reid') {
                refreshData();
            }
        };
        window.addEventListener('trigger-refresh', handleRefresh as EventListener);
        return () => window.removeEventListener('trigger-refresh', handleRefresh as EventListener);
    }, [refreshData]);

    const loadImageByPath = async (path: string, setFn: React.Dispatch<React.SetStateAction<Map<string, string>>>) => {
        try {
            const response = await window.api.viewImage(path);
            if (response.ok && response.data) {
                const blob = new Blob([response.data as unknown as BlobPart], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                setFn(prev => new Map(prev).set(path, url));
            }
        } catch (e) { console.error('Failed to load image:', path, e); }
    };

    const loadImagesForIndividuals = (inds: ReidIndividual[]) => {
        for (const ind of inds) {
            for (const det of ind.detections) {
                const path = det.image_preview_path || det.image_path;
                if (path && !imageUrls.has(path)) loadImageByPath(path, setImageUrls);
            }
        }
    };

    // Initial load - fetch runs and first page of each
    useEffect(() => {
        const loadData = async () => {
            setLoading(true);
            try {
                const runsRes = await window.api.getReidRuns();
                if (runsRes.ok && runsRes.runs) {
                    // Sort runs by created_at descending (new to old)
                    const sortedRuns = runsRes.runs.sort((a, b) => b.created_at - a.created_at);
                    setRuns(sortedRuns);
                    const newIndividuals = new Map<number, ReidIndividual[]>();
                    const newPagination = new Map<number, { page: number; hasMore: boolean }>();
                    
                    for (const run of sortedRuns) {
                        const res = await window.api.getReidResults({ runId: run.id, page: 1, pageSize: PAGE_SIZE });
                        if (res.ok && res.result) {
                            newIndividuals.set(run.id, res.result.individuals);
                            newPagination.set(run.id, { page: 1, hasMore: res.result.pagination.has_more });
                            loadImagesForIndividuals(res.result.individuals);
                        }
                    }
                    setIndividuals(newIndividuals);
                    setPagination(newPagination);
                }
            } catch (e) { console.error('Failed to load ReID data:', e); }
            setLoading(false);
        };
        loadData();
    }, [refreshTrigger]);

    // Load more individuals for a specific run
    const loadMoreForRun = async (runId: number) => {
        const currentPagination = pagination.get(runId);
        if (!currentPagination || !currentPagination.hasMore || loadingMore.get(runId)) return;

        setLoadingMore(prev => new Map(prev).set(runId, true));
        
        try {
            const nextPage = currentPagination.page + 1;
            const res = await window.api.getReidResults({ runId, page: nextPage, pageSize: PAGE_SIZE });
            
            if (res.ok && res.result) {
                const result = res.result;
                setIndividuals(prev => {
                    const existing = prev.get(runId) || [];
                    return new Map(prev).set(runId, [...existing, ...result.individuals]);
                });
                setPagination(prev => new Map(prev).set(runId, { page: nextPage, hasMore: result.pagination.has_more }));
                loadImagesForIndividuals(res.result.individuals);
            }
        } catch (e) { console.error('Failed to load more:', e); }
        
        setLoadingMore(prev => new Map(prev).set(runId, false));
    };

    const handleMenuOpen = (e: React.MouseEvent<HTMLElement>, runId: number) => { setMenuAnchor(e.currentTarget); setSelectedRunId(runId); };
    const handleMenuClose = () => { setMenuAnchor(null); setSelectedRunId(null); };
    const handleRename = () => { const run = runs.find(r => r.id === selectedRunId); if (run) { setRunToRename({ id: run.id, name: run.name }); setRenameDialogOpen(true); } handleMenuClose(); };
    const handleConfirmRename = async (newName: string) => { if (runToRename) { await window.api.updateReidRunName(runToRename.id, newName); setRefreshTrigger(t => t + 1); } setRenameDialogOpen(false); setRunToRename(null); };
    const handleDelete = async () => { if (selectedRunId && window.confirm('Delete this ReID run?')) { await window.api.deleteReidRun(selectedRunId); setRefreshTrigger(t => t + 1); } handleMenuClose(); };
    const handleIndividualClick = (ind: ReidIndividual) => { setSelectedIndividual(ind); };

    if (loading) return (
        <Box sx={{ pt: '64px', px: 3, pb: 3, minHeight: '100vh' }}>
            {/* Header skeleton */}
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', py: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Skeleton variant="circular" width={28} height={28} />
                    <Skeleton variant="text" sx={{ fontSize: '1.5rem', width: 180 }} />
                    <Skeleton variant="text" sx={{ fontSize: '0.875rem', width: 60 }} />
                </Box>
                <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                    <Skeleton variant="rounded" width={36} height={36} sx={{ borderRadius: 2 }} />
                    <Skeleton variant="circular" width={36} height={36} />
                </Box>
            </Box>
            
            {/* Run group skeleton */}
            {[1, 2].map((groupIdx) => (
                <Box key={groupIdx} sx={{ mb: 3 }}>
                    {/* Run header skeleton */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2, pl: 1 }}>
                        <Skeleton variant="circular" width={20} height={20} />
                        <Skeleton variant="text" sx={{ fontSize: '1.1rem', width: 200 }} />
                        <Skeleton variant="rounded" width={80} height={24} sx={{ borderRadius: 1 }} />
                        <Skeleton variant="rounded" width={100} height={24} sx={{ borderRadius: 1 }} />
                    </Box>
                    
                    {/* Individual cards skeleton */}
                    <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 2, pl: 4 }}>
                        {[...Array(6)].map((_, i) => (
                            <Box key={i} sx={{ borderRadius: 2, overflow: 'hidden', border: `1px solid ${theme.palette.divider}` }}>
                                <Skeleton variant="rectangular" width="100%" height={130} animation="wave" />
                                <Box sx={{ p: 1.25 }}>
                                    <Skeleton variant="text" width="70%" height={20} animation="wave" />
                                    <Skeleton variant="text" width="50%" height={16} animation="wave" />
                                </Box>
                            </Box>
                        ))}
                    </Box>
                </Box>
            ))}
        </Box>
    );

    // If viewing an individual, show the detail view (full height, navbar spacer in header)
    if (selectedIndividual) {
        return (
            <Box sx={{ height: '100vh', overflow: 'hidden' }}>
                <IndividualDetailView
                    individual={selectedIndividual}
                    onBack={() => setSelectedIndividual(null)}
                />
            </Box>
        );
    }

    return (
        <Box sx={{ pt: '64px', px: 3, pb: 3, minHeight: '100vh' }}>
            <RefreshNotification 
                watchJobTypes={['reid']}
                onRefresh={refreshData}
                message="Re-identification completed"
            />
            {runs.length === 0 ? (
                <Box sx={{ height: 'calc(100vh - 180px)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 2, opacity: 0.6 }}>
                    <Fingerprint size={64} weight="thin" color={theme.palette.text.primary} />
                    <Typography variant="h5" fontWeight="500" color="text.primary">No ReID runs yet</Typography>
                    <Typography variant="body1" color="text.secondary">Run Re-identification from the Library or Classification page</Typography>
                    <Button
                        variant="contained"
                        startIcon={<Sparkle size={18} />}
                        onClick={() => navigate('/classification')}
                        sx={{
                            mt: 2,
                            borderRadius: 2,
                            textTransform: 'none',
                            bgcolor: theme.palette.mode === 'dark' ? '#FFFFFF' : '#000000',
                            color: theme.palette.mode === 'dark' ? '#000000' : '#FFFFFF',
                            '&:hover': {
                                bgcolor: theme.palette.mode === 'dark' ? '#E0E0E0' : '#333333'
                            }
                        }}
                    >
                        Go to Classification
                    </Button>
                </Box>
            ) : (
                <Box>
                    {/* Header - only shown when there's data */}
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', py: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <Fingerprint size={28} weight="duotone" />
                            <Typography variant="h5" fontWeight={600}>Re-identification</Typography>
                            <Typography variant="body2" color="text.secondary">{runs.length} run{runs.length !== 1 ? 's' : ''}</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                            <LibrarySearchBar value={searchQuery} onSearch={setSearchQuery} />
                            
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
                        </Box>
                    </Box>
                    {runs.filter(run => {
                        if (!searchQuery) return true;
                        const q = searchQuery.toLowerCase();
                        const runIndividuals = individuals.get(run.id) || [];
                        return run.name.toLowerCase().includes(q) ||
                               run.species.toLowerCase().includes(q) ||
                               runIndividuals.some(ind => ind.display_name.toLowerCase().includes(q));
                    }).map(run => {
                        const runIndividuals = individuals.get(run.id) || [];
                        const runPagination = pagination.get(run.id);
                        const isLoadingMore = loadingMore.get(run.id) || false;
                        return (
                            <RunGroup 
                                key={run.id} 
                                run={run} 
                                individuals={runIndividuals} 
                                imageUrls={imageUrls} 
                                onIndividualClick={handleIndividualClick} 
                                onMenuOpen={handleMenuOpen}
                                hasMore={runPagination?.hasMore || false}
                                loadingMore={isLoadingMore}
                                onLoadMore={() => loadMoreForRun(run.id)}
                            />
                        );
                    })}
                </Box>
            )}
            <Menu anchorEl={menuAnchor} open={Boolean(menuAnchor)} onClose={handleMenuClose} PaperProps={{ sx: { borderRadius: 2, minWidth: 160 } }}>
                <MenuItem onClick={handleRename}><PencilSimple size={18} style={{ marginRight: 8 }} /> Rename</MenuItem>
                <MenuItem onClick={handleDelete} sx={{ color: 'error.main' }}><Trash size={18} style={{ marginRight: 8 }} /> Delete</MenuItem>
            </Menu>
            <GroupNameDialog open={renameDialogOpen} onClose={() => { setRenameDialogOpen(false); setRunToRename(null); }} onConfirm={handleConfirmRename} title="Rename ReID Run" initialValue={runToRename?.name || ''} />
            
            {/* Settings Menu */}
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
                            minWidth: '220px',
                            p: 2,
                            mt: 1
                        }
                    }
                }}
            >
                <Typography variant="subtitle2" fontWeight="600" sx={{ mb: 1.5 }}>
                    Display Settings
                </Typography>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.5 }}>
                    <Typography variant="body2">
                        Liquid Glass BBox
                    </Typography>
                    <Switch
                        size="small"
                        checked={useLiquidGlass}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                            setUseLiquidGlass(e.target.checked);
                            localStorage.setItem('mediaExplorer_useLiquidGlass', e.target.checked.toString());
                        }}
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
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                                setUseRayTracedGlass(e.target.checked);
                                localStorage.setItem('mediaExplorer_useRayTracedGlass', e.target.checked.toString());
                            }}
                        />
                    </Box>
                )}
            </Menu>

            {/* Floating Action Button - Back to Top */}
            <Box
                sx={{
                    position: 'fixed',
                    top: 80,
                    right: 16,
                    zIndex: 1000,
                    p: 1.5,
                    opacity: isScrolled ? 1 : 0,
                    transform: isScrolled ? 'translateX(0)' : 'translateX(calc(100% + 32px))',
                    transition: 'transform 0.5s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.3s ease',
                    pointerEvents: isScrolled ? 'auto' : 'none',
                    '&::before': {
                        content: '""',
                        position: 'absolute',
                        inset: 0,
                        borderRadius: '24px',
                        background: 'rgba(0,0,0,0.35)',
                        filter: 'blur(30px)',
                        zIndex: -1,
                        pointerEvents: 'none'
                    }
                }}
            >
                <Tooltip title="Back to Top">
                    <span>
                        <LiquidGlassButton
                            size={32}
                            icon={<ArrowLineUp size={16} />}
                            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                        />
                    </span>
                </Tooltip>
            </Box>
        </Box>
    );
};

export default ReIDPage;
