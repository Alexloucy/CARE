import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useOutletContext, useNavigate } from 'react-router-dom';
import {
    Box, Typography, CircularProgress, IconButton, Menu, MenuItem,
    Chip, alpha, useTheme, Collapse, Modal, Backdrop, Fade, Skeleton, Button,
    Switch, Tooltip
} from '@mui/material';
import {
    ArrowLineUp, Fingerprint, DotsThreeVertical, PencilSimple, Trash, CaretDown, CaretRight,
    Images as ImagesIcon, X, CaretLeft, MagnifyingGlassPlus, MagnifyingGlassMinus, Sparkle,
    Gear
} from '@phosphor-icons/react';
import { LiquidGlassButton } from '../../components/LiquidGlassButton';
import { GroupNameDialog } from '../../components/GroupNameDialog';
import { LibrarySearchBar } from '../../components/library/LibrarySearchBar';
import { DetectionBox } from '../../components/ImageModal';
import { LiquidGlassOverlay } from '../../components/LiquidGlassOverlay';
import { RefreshNotification } from '../../components/RefreshNotification';
import { Detection } from '../../types/electron';

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

const IndividualModal: React.FC<{ open: boolean; onClose: () => void; individual: ReidIndividual | null; imageUrls: Map<string, string>; fullImageUrls: Map<string, string>; loadFullImage: (path: string) => void; useLiquidGlass?: boolean; useRayTracedGlass?: boolean }> = ({ open, onClose, individual, imageUrls, fullImageUrls, loadFullImage, useLiquidGlass = true, useRayTracedGlass = true }) => {
    const theme = useTheme();
    const [currentIndex, setCurrentIndex] = useState(0);
    const [zoom, setZoom] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const dragStart = useRef({ x: 0, y: 0 });
    const imageRef = useRef<HTMLImageElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [imgDims, setImgDims] = useState({ natural: { width: 0, height: 0 }, displayed: { width: 0, height: 0 }, container: { width: 0, height: 0 } });

    const detections = individual?.detections || [];
    const currentDet = detections[currentIndex];
    const currentUrl = currentDet ? (fullImageUrls.get(currentDet.image_path) || imageUrls.get(currentDet.image_path)) : undefined;

    // Preload all images for this individual when modal opens
    useEffect(() => {
        if (open && individual && detections.length > 0) {
            // Preload all images for smoother navigation
            detections.forEach(det => {
                if (!fullImageUrls.has(det.image_path)) {
                    loadFullImage(det.image_path);
                }
            });
        }
    }, [open, individual?.id]);

    useEffect(() => { if (open && currentDet) { setZoom(1); setPosition({ x: 0, y: 0 }); } }, [open, currentIndex, currentDet]);
    useEffect(() => { setCurrentIndex(0); }, [individual?.id]);

    const handleImageLoad = () => {
        if (imageRef.current && containerRef.current) {
            const img = imageRef.current, rect = containerRef.current.getBoundingClientRect();
            const aspect = img.naturalWidth / img.naturalHeight, cAspect = rect.width / rect.height;
            const [dw, dh] = aspect > cAspect ? [rect.width, rect.width / aspect] : [rect.height * aspect, rect.height];
            setImgDims({ natural: { width: img.naturalWidth, height: img.naturalHeight }, displayed: { width: dw, height: dh }, container: { width: rect.width, height: rect.height } });
        }
    };

    const transformBbox = () => {
        if (!containerRef.current || imgDims.natural.width === 0 || !currentDet) return null;
        const scale = imgDims.displayed.width / imgDims.natural.width;
        const rect = containerRef.current.getBoundingClientRect();
        const offsetX = (rect.width - imgDims.displayed.width) / 2, offsetY = (rect.height - imgDims.displayed.height) / 2;
        return { x: offsetX + currentDet.x1 * scale, y: offsetY + currentDet.y1 * scale, width: (currentDet.x2 - currentDet.x1) * scale, height: (currentDet.y2 - currentDet.y1) * scale };
    };

    if (!individual) return null;
    const bbox = transformBbox();

    return (
        <Modal open={open} onClose={onClose} closeAfterTransition slots={{ backdrop: Backdrop }} slotProps={{ backdrop: { timeout: 500, sx: { backgroundColor: 'rgba(0,0,0,0.85)' } } }}>
            <Fade in={open}>
                <Box onClick={(e: React.MouseEvent) => e.stopPropagation()} sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '90vw', height: '90vh', bgcolor: 'background.paper', borderRadius: 4, overflow: 'hidden', boxShadow: 24, display: 'flex', outline: 'none' }}>
                    {/* SVG Filter for Liquid Glass Effect */}
                    <svg style={{ display: 'none' }}>
                        <filter id="container-glass" x="0%" y="0%" width="100%" height="100%">
                            <feTurbulence type="fractalNoise" baseFrequency="0.008 0.008" numOctaves="2" seed="92" result="noise" />
                            <feGaussianBlur in="noise" stdDeviation="0.02" result="blur" />
                            <feDisplacementMap in="SourceGraphic" in2="blur" scale="77" xChannelSelector="R" yChannelSelector="G" />
                        </filter>
                    </svg>
                    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
                        <Box ref={containerRef} sx={{ flex: 1, position: 'relative', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: 'black', cursor: isDragging ? 'grabbing' : 'grab' }}
                            onWheel={(e: React.WheelEvent) => { e.stopPropagation(); setZoom(z => e.deltaY < 0 ? Math.min(z + 0.1, 5) : Math.max(z - 0.1, 0.5)); }}
                            onMouseDown={(e: React.MouseEvent) => { setIsDragging(true); dragStart.current = { x: e.clientX - position.x, y: e.clientY - position.y }; }}
                            onMouseMove={(e: React.MouseEvent) => { if (isDragging) { e.preventDefault(); setPosition({ x: e.clientX - dragStart.current.x, y: e.clientY - dragStart.current.y }); } }}
                            onMouseUp={() => setIsDragging(false)} onMouseLeave={() => setIsDragging(false)}>
                            {currentUrl ? (
                                <>
                                    <img ref={imageRef} src={currentUrl} alt={individual.display_name} onLoad={handleImageLoad} style={{ width: '100%', height: '100%', objectFit: 'contain', transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`, transition: isDragging ? 'none' : 'transform 0.1s', userSelect: 'none' }} draggable={false} />
                                    {bbox && imgDims.displayed.width > 0 && (
                                        useLiquidGlass && useRayTracedGlass && imgDims.container.width > 0 ? (
                                            /* Ray-traced liquid glass - positioned over displayed image area */
                                            <Box sx={{ 
                                                position: 'absolute', 
                                                top: (imgDims.container.height - imgDims.displayed.height) / 2,
                                                left: (imgDims.container.width - imgDims.displayed.width) / 2,
                                                width: imgDims.displayed.width, 
                                                height: imgDims.displayed.height, 
                                                pointerEvents: 'none', 
                                                transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`, 
                                                transition: isDragging ? 'none' : 'transform 0.1s' 
                                            }}>
                                                <LiquidGlassOverlay
                                                    imageUrl={currentUrl}
                                                    bboxes={[{ 
                                                        bbox: {
                                                            // Adjust bbox coordinates relative to displayed image (remove offset)
                                                            x: bbox.x - (imgDims.container.width - imgDims.displayed.width) / 2,
                                                            y: bbox.y - (imgDims.container.height - imgDims.displayed.height) / 2,
                                                            width: bbox.width,
                                                            height: bbox.height
                                                        }, 
                                                        label: individual.display_name, 
                                                        detection: { ...currentDet, label: individual.display_name } as Detection 
                                                    }]}
                                                    containerWidth={imgDims.displayed.width}
                                                    containerHeight={imgDims.displayed.height}
                                                    customPopupContent={
                                                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                                                                <Fingerprint size={18} weight="fill" color={individual.color} />
                                                                <Typography variant="subtitle2" fontWeight="700">Individual Details</Typography>
                                                            </Box>
                                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                                <Typography variant="caption" color="text.secondary">Individual</Typography>
                                                                <Box sx={{ bgcolor: alpha(individual.color, 0.15), color: individual.color, px: 1, py: 0.2, borderRadius: 1, fontSize: '0.75rem', fontWeight: 600 }}>
                                                                    {individual.display_name}
                                                                </Box>
                                                            </Box>
                                                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                                <Typography variant="caption" color="text.secondary">Sightings</Typography>
                                                                <Typography variant="caption" fontWeight="600">{individual.member_count}</Typography>
                                                            </Box>
                                                        </Box>
                                                    }
                                                />
                                            </Box>
                                        ) : (
                                            /* CSS-based detection box (liquid glass or classic) */
                                            <Box sx={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`, transition: isDragging ? 'none' : 'transform 0.1s' }}>
                                                <DetectionBox
                                                    bbox={bbox}
                                                    detection={{ ...currentDet, label: individual.display_name } as Detection}
                                                    zoom={zoom}
                                                    containerWidth={imgDims.displayed.width}
                                                    containerHeight={imgDims.displayed.height}
                                                    useLiquidGlass={useLiquidGlass}
                                                    popupTitle="Individual Details"
                                                    popupIcon={<Fingerprint size={18} weight="fill" color={individual.color} />}
                                                    customPopupContent={
                                                        <>
                                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                                                                <Fingerprint size={18} weight="fill" color={individual.color} />
                                                                <Typography variant="subtitle2" fontWeight="700">Individual Details</Typography>
                                                            </Box>
                                                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                                    <Typography variant="caption" color="text.secondary">Individual</Typography>
                                                                    <Box sx={{ bgcolor: alpha(individual.color, 0.15), color: individual.color, px: 1, py: 0.2, borderRadius: 1, fontSize: '0.75rem', fontWeight: 600 }}>
                                                                        {individual.display_name}
                                                                    </Box>
                                                                </Box>
                                                                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                                                    <Typography variant="caption" color="text.secondary">Sightings</Typography>
                                                                    <Typography variant="caption" fontWeight="600">{individual.member_count}</Typography>
                                                                </Box>
                                                            </Box>
                                                        </>
                                                    }
                                                />
                                            </Box>
                                        )
                                    )}
                                </>
                            ) : <CircularProgress />}
                            {currentIndex > 0 && <IconButton onClick={() => setCurrentIndex(i => i - 1)} sx={{ position: 'absolute', left: 16, top: '50%', transform: 'translateY(-50%)', color: 'white', bgcolor: 'rgba(0,0,0,0.4)', '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' } }}><CaretLeft size={32} /></IconButton>}
                            {currentIndex < detections.length - 1 && <IconButton onClick={() => setCurrentIndex(i => i + 1)} sx={{ position: 'absolute', right: 16, top: '50%', transform: 'translateY(-50%)', color: 'white', bgcolor: 'rgba(0,0,0,0.4)', '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' } }}><CaretRight size={32} /></IconButton>}
                            <Box sx={{ position: 'absolute', top: 16, right: 16, display: 'flex', gap: 1, bgcolor: 'rgba(0,0,0,0.4)', borderRadius: 3, p: 0.5 }}>
                                <IconButton onClick={() => setZoom(z => Math.max(z - 0.5, 0.5))} size="small" sx={{ color: 'white' }}><MagnifyingGlassMinus size={20} /></IconButton>
                                <IconButton onClick={() => setZoom(z => Math.min(z + 0.5, 5))} size="small" sx={{ color: 'white' }}><MagnifyingGlassPlus size={20} /></IconButton>
                                <IconButton onClick={onClose} size="small" sx={{ color: 'white' }}><X size={20} /></IconButton>
                            </Box>
                            <Box sx={{ position: 'absolute', bottom: 16, left: '50%', transform: 'translateX(-50%)', bgcolor: 'rgba(0,0,0,0.6)', color: 'white', px: 2, py: 0.5, borderRadius: 2, fontSize: '0.875rem' }}>{currentIndex + 1} / {detections.length}</Box>
                        </Box>
                    </Box>
                    <Box sx={{ width: 320, borderLeft: `1px solid ${theme.palette.divider}`, display: 'flex', flexDirection: 'column' }}>
                        <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1 }}><Box sx={{ width: 16, height: 16, borderRadius: '50%', bgcolor: individual.color }} /><Typography variant="h6" fontWeight={600}>{individual.display_name}</Typography></Box>
                            <Typography variant="body2" color="text.secondary">{individual.member_count} sighting{individual.member_count !== 1 ? 's' : ''}</Typography>
                        </Box>
                        <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
                            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1.5 }}>All Sightings</Typography>
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                {detections.map((det, idx) => (
                                    <Box key={det.id} onClick={() => setCurrentIndex(idx)} sx={{ display: 'flex', gap: 1.5, p: 1, borderRadius: 2, cursor: 'pointer', bgcolor: idx === currentIndex ? alpha(individual.color, 0.15) : 'transparent', border: `1px solid ${idx === currentIndex ? individual.color : 'transparent'}`, '&:hover': { bgcolor: alpha(individual.color, 0.1) } }}>
                                        <Box sx={{ width: 56, height: 56, borderRadius: 1.5, overflow: 'hidden', bgcolor: theme.palette.mode === 'light' ? '#f0f0f0' : '#2a2a2a', flexShrink: 0 }}>
                                            {imageUrls.get(det.image_preview_path || det.image_path) ? <Box component="img" src={imageUrls.get(det.image_preview_path || det.image_path)} sx={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : <Box sx={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Fingerprint size={24} weight="thin" /></Box>}
                                        </Box>
                                        <Box sx={{ flex: 1, minWidth: 0 }}>
                                            <Typography variant="caption" noWrap sx={{ display: 'block' }}>{det.image_path.split(/[/\\]/).pop()}</Typography>
                                            <Typography variant="caption" color="text.secondary">{det.label}</Typography>
                                        </Box>
                                    </Box>
                                ))}
                            </Box>
                        </Box>
                    </Box>
                </Box>
            </Fade>
        </Modal>
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
    const [modalOpen, setModalOpen] = useState(false);
    const [imageUrls, setImageUrls] = useState<Map<string, string>>(new Map());
    const [fullImageUrls, setFullImageUrls] = useState<Map<string, string>>(new Map());
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
                    setRuns(runsRes.runs);
                    const newIndividuals = new Map<number, ReidIndividual[]>();
                    const newPagination = new Map<number, { page: number; hasMore: boolean }>();
                    
                    for (const run of runsRes.runs) {
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

    const loadFullImage = (path: string) => {
        if (!fullImageUrls.has(path)) {
            loadImageByPath(path, setFullImageUrls);
        }
    };

    const handleMenuOpen = (e: React.MouseEvent<HTMLElement>, runId: number) => { setMenuAnchor(e.currentTarget); setSelectedRunId(runId); };
    const handleMenuClose = () => { setMenuAnchor(null); setSelectedRunId(null); };
    const handleRename = () => { const run = runs.find(r => r.id === selectedRunId); if (run) { setRunToRename({ id: run.id, name: run.name }); setRenameDialogOpen(true); } handleMenuClose(); };
    const handleConfirmRename = async (newName: string) => { if (runToRename) { await window.api.updateReidRunName(runToRename.id, newName); setRefreshTrigger(t => t + 1); } setRenameDialogOpen(false); setRunToRename(null); };
    const handleDelete = async () => { if (selectedRunId && window.confirm('Delete this ReID run?')) { await window.api.deleteReidRun(selectedRunId); setRefreshTrigger(t => t + 1); } handleMenuClose(); };
    const handleIndividualClick = (ind: ReidIndividual) => { setSelectedIndividual(ind); setModalOpen(true); };

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
                            <LibrarySearchBar onSearch={setSearchQuery} />
                            
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
            
            <IndividualModal open={modalOpen} onClose={() => setModalOpen(false)} individual={selectedIndividual} imageUrls={imageUrls} fullImageUrls={fullImageUrls} loadFullImage={loadFullImage} useLiquidGlass={useLiquidGlass} useRayTracedGlass={useRayTracedGlass} />

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
