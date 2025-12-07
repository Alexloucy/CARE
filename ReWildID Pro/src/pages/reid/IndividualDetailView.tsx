import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
    Box, Typography, IconButton, Menu,
    useTheme, alpha, Tooltip, Slider, Divider,
    ToggleButton, ToggleButtonGroup, Switch
} from '@mui/material';
import { Virtuoso } from 'react-virtuoso';
import {
    CaretLeft, Gear, ArrowCounterClockwise, Fingerprint
} from '@phosphor-icons/react';
import ImageModal from '../../components/ImageModal';
import { Detection, DBImage } from '../../types/electron';
import { ReidIndividual } from './types';

const IndividualDetailView: React.FC = () => {
    const theme = useTheme();
    const navigate = useNavigate();
    const location = useLocation();

    // Try to get individual from state, otherwise we might need to fetch it (not implemented yet)
    const individual = location.state?.individual as ReidIndividual | undefined;

    const [loading, setLoading] = useState(true);
    const [dbImages, setDbImages] = useState<DBImage[]>([]);
    const [imageUrls, setImageUrls] = useState<Record<number, string>>({});
    const [fullImageUrls, setFullImageUrls] = useState<Record<number, string>>({});
    const [selectedImage, setSelectedImage] = useState<DBImage | null>(null);
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
    const [useLiquidGlass] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_useLiquidGlass');
        return saved === null ? true : saved === 'true';
    });
    const [useRayTracedGlass] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_useRayTracedGlass');
        return saved === null ? true : saved === 'true';
    });
    const [showBoundingBoxes, setShowBoundingBoxes] = useState(() => {
        const saved = localStorage.getItem('mediaExplorer_showBoundingBoxes');
        return saved === null ? true : saved === 'true';
    });
    const [showColors, setShowColors] = useState(() => {
        const saved = localStorage.getItem('reid_showColors');
        return saved === 'true'; // default false
    });
    const [containerWidth, setContainerWidth] = useState(0);
    const containerRef = useRef<HTMLDivElement>(null);
    const [settingsMenuPos, setSettingsMenuPos] = useState<{ top: number; left: number } | null>(null);

    // If no individual found (e.g. direct link without state), redirect back for now
    useEffect(() => {
        if (!individual) {
            navigate('/reid');
        }
    }, [individual, navigate]);

    // Persist settings similar to MediaExplorer
    useEffect(() => {
        localStorage.setItem('mediaExplorer_gridSize', gridItemSize.toString());
        localStorage.setItem('mediaExplorer_showNames', showFileNames.toString());
        localStorage.setItem('mediaExplorer_aspectRatio', aspectRatio);
        localStorage.setItem('mediaExplorer_showBoundingBoxes', showBoundingBoxes.toString());
    }, [gridItemSize, showFileNames, aspectRatio, showBoundingBoxes]);

    // Listen for settings updates triggered elsewhere (Settings page / MediaExplorer)
    useEffect(() => {
        const handleStorageChange = (e: StorageEvent) => {
            if (!e.key?.startsWith('mediaExplorer_') || e.newValue === null) return;
            switch (e.key) {
                case 'mediaExplorer_gridSize':
                    setGridItemSize(parseInt(e.newValue, 10));
                    break;
                case 'mediaExplorer_showNames':
                    setShowFileNames(e.newValue === 'true');
                    break;
                case 'mediaExplorer_aspectRatio':
                    setAspectRatio(e.newValue);
                    break;
                case 'mediaExplorer_showBoundingBoxes':
                    setShowBoundingBoxes(e.newValue === 'true');
                    break;
            }
            if (e.key === 'reid_showColors' && e.newValue !== null) {
                setShowColors(e.newValue === 'true');
            }
        };
        window.addEventListener('storage', handleStorageChange);
        return () => window.removeEventListener('storage', handleStorageChange);
    }, []);

    // Zoom Handler
    const zoomContainerRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        const node = zoomContainerRef.current;
        if (!node) return;

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

        node.addEventListener('wheel', handleWheel, { passive: false });
        return () => {
            node.removeEventListener('wheel', handleWheel);
        };
    }, [loading]);

    // Fetch DBImage objects for this individual's detections
    useEffect(() => {
        if (!individual) return;
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
    }, [individual]);

    // Measure container width
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


    // Column calculation
    const horizontalPadding = 64;
    const gap = 16;
    const availableWidth = containerWidth - horizontalPadding;
    const columns = availableWidth > 0 ? Math.max(1, Math.floor((availableWidth + gap) / (gridItemSize + gap))) : 1;
    const actualItemWidth = columns > 0 ? (availableWidth - (gap * (columns - 1))) / columns : gridItemSize;

    const getRowHeight = useCallback(() => {
        const [wRaw, hRaw] = aspectRatio.split('/').map(Number);
        const w = isNaN(wRaw) || wRaw === 0 ? 1 : wRaw;
        const h = isNaN(hRaw) ? 1 : hRaw;
        return actualItemWidth * (h / w);
    }, [aspectRatio, actualItemWidth]);

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
        if (!individual) return [];
        return individual.detections
            .filter(d => d.image_id === imgId)
            .map(d => ({
                id: d.id,
                image_id: d.image_id,
                label: d.label, // Keep the actual species label, NOT the individual name
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

    const handleImageClick = useCallback((img: DBImage) => {
        setSelectedImage(img);
        loadFullImage(img);
    }, [loadFullImage]);

    // Preload nearby images
    useEffect(() => {
        if (selectedImage && currentIndex !== -1) {
            for (let offset = 1; offset <= 3; offset++) {
                if (currentIndex + offset < dbImages.length) loadFullImage(dbImages[currentIndex + offset]);
                if (currentIndex - offset >= 0) loadFullImage(dbImages[currentIndex - offset]);
            }
        }
    }, [selectedImage, currentIndex, dbImages, loadFullImage]);

    // Header component
    const headerContent = useMemo(() => (
        <Box>
            <Box sx={{ height: 64 }} />
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 2, px: 4 }}>
                <IconButton onClick={() => navigate(-1)} sx={{ bgcolor: alpha(theme.palette.text.primary, 0.05), '&:hover': { bgcolor: alpha(theme.palette.text.primary, 0.1) } }}>
                    <CaretLeft size={20} />
                </IconButton>
                {showColors && individual && <Box sx={{ width: 20, height: 20, borderRadius: '50%', bgcolor: individual.color, border: '2px solid', borderColor: theme.palette.background.paper, boxShadow: 1 }} />}
                <Box sx={{ flex: 1 }}>
                    <Typography variant="h5" fontWeight={600}>{individual?.display_name}</Typography>
                    <Typography variant="body2" color="text.secondary">
                        {individual?.member_count} sighting{individual?.member_count !== 1 ? 's' : ''} • {dbImages.length} image{dbImages.length !== 1 ? 's' : ''}
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
    ), [individual, dbImages.length, navigate, theme, setSettingsMenuPos, showColors]);

    const virtuosoComponents = useMemo(() => ({
        Header: () => headerContent
    }), [headerContent]);

    if (!individual) return null; // Should redirect via useEffect

    if (loading) {
        return (
            <Box sx={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', pt: '64px' }}>
                <Typography color="text.secondary">Loading images...</Typography>
            </Box>
        );
    }

    const rowHeight = getRowHeight() + 24;

    return (
        <Box sx={{ height: '100vh', overflow: 'hidden' }}>
            <Box
                ref={zoomContainerRef}
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
                                        const fileName = img.original_path.split(/[\\/]/).pop() || 'image';

                                        return (
                                            <Box
                                                key={img.id}
                                                onClick={() => handleImageClick(img)}
                                                onDragStart={(e: React.DragEvent) => e.preventDefault()}
                                                sx={{
                                                    minWidth: 0,
                                                    cursor: 'pointer',
                                                    userSelect: 'none'
                                                }}
                                            >
                                                <Box
                                                    sx={{
                                                        width: '100%',
                                                        aspectRatio,
                                                        borderRadius: 3,
                                                        overflow: 'hidden',
                                                        position: 'relative',
                                                        bgcolor: theme.palette.mode === 'light' ? '#f5f5f5' : '#1a1a1a',
                                                        transition: 'all 0.15s ease-in-out',
                                                        '&:hover': {
                                                            transform: 'translateY(-2px)',
                                                            boxShadow: '0 8px 24px rgba(0,0,0,0.2)'
                                                        },
                                                        '&:hover .file-overlay': {
                                                            opacity: 1
                                                        }
                                                    }}
                                                >
                                                    {url ? (
                                                        <Box
                                                            component="img"
                                                            src={url}
                                                            draggable={false}
                                                            sx={{
                                                                width: '100%',
                                                                height: '100%',
                                                                objectFit: 'cover',
                                                                userSelect: 'none',
                                                                WebkitUserDrag: 'none',
                                                                pointerEvents: 'none'
                                                            }}
                                                        />
                                                    ) : (
                                                        <Box sx={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                                            <Fingerprint size={32} weight="thin" color={theme.palette.text.disabled} />
                                                        </Box>
                                                    )}

                                                    <Box className="file-overlay" sx={{
                                                        position: 'absolute',
                                                        bottom: 0,
                                                        left: 0,
                                                        right: 0,
                                                        p: 1.5,
                                                        background: 'linear-gradient(to top, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0) 100%)',
                                                        color: 'white',
                                                        pointerEvents: 'none',
                                                        opacity: showFileNames ? 1 : 0,
                                                        transition: 'opacity 0.3s ease'
                                                    }}>
                                                        <Typography variant="body2" noWrap sx={{ fontWeight: 500, textShadow: '0 1px 2px rgba(0,0,0,0.5)' }}>
                                                            {fileName}
                                                        </Typography>
                                                    </Box>
                                                </Box>
                                            </Box>
                                        );
                                    })}
                                </Box>
                            );
                        }}
                    />
                </Box>
            </Box>

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
                </ToggleButtonGroup>
                <Divider sx={{ my: 1 }} />
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.5 }}>
                    <Typography variant="subtitle2" fontWeight="600">Show File Names</Typography>
                    <Switch size="small" checked={showFileNames} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setShowFileNames(e.target.checked)} />
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.5 }}>
                    <Typography variant="subtitle2" fontWeight="600">Show Bounding Boxes</Typography>
                    <Switch size="small" checked={showBoundingBoxes} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setShowBoundingBoxes(e.target.checked)} />
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.5 }}>
                    <Typography variant="subtitle2" fontWeight="600">Show ID Colors</Typography>
                    <Switch size="small" checked={showColors} onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setShowColors(e.target.checked); localStorage.setItem('reid_showColors', e.target.checked.toString()); }} />
                </Box>
            </Menu>

            {/* Image Modal */}
            {(() => {
                const isOpen = selectedImage !== null;
                const imageUrl = selectedImage ? (fullImageUrls[selectedImage.id] || imageUrls[selectedImage.id]) : undefined;
                const dets = selectedImage ? getDetectionsForImage(selectedImage.id) : [];
                const file = selectedImage ? {
                    name: selectedImage.original_path.split('\\').pop() || selectedImage.original_path.split('/').pop() || 'unknown',
                    isDirectory: false,
                    path: selectedImage.original_path
                } : undefined;
                // Build reidResults from current individual context
                const reidInfo = individual ? [{
                    individualId: individual.id,
                    individualName: individual.name,
                    individualDisplayName: individual.display_name,
                    individualColor: individual.color,
                    runId: individual.run_id,
                    runName: '', // We don't have run name in the individual object 
                    species: dets[0]?.label || ''
                }] : [];
                return (
                    <ImageModal
                        open={isOpen}
                        onClose={() => setSelectedImage(null)}
                        imageUrl={imageUrl}
                        imageId={selectedImage?.id}
                        file={file}
                        detections={showBoundingBoxes ? dets : []}
                        reidResults={reidInfo}
                        onNext={hasNext ? goNext : undefined}
                        onPrev={hasPrev ? goPrev : undefined}
                        hasNext={hasNext}
                        hasPrev={hasPrev}
                        useLiquidGlass={useLiquidGlass}
                        useRayTracedGlass={useRayTracedGlass}
                    />
                );
            })()}
        </Box>
    );
};

export default IndividualDetailView;
