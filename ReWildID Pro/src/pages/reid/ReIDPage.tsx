import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useOutletContext, useNavigate } from 'react-router-dom';
import {
    Box, Typography, CircularProgress, IconButton, Menu, MenuItem,
    Chip, alpha, useTheme, Collapse, Modal, Backdrop, Fade, Skeleton, Button
} from '@mui/material';
import {
    Fingerprint, DotsThreeVertical, PencilSimple, Trash, CaretDown, CaretRight,
    Images as ImagesIcon, X, CaretLeft, MagnifyingGlassPlus, MagnifyingGlassMinus, Sparkle
} from '@phosphor-icons/react';
import { GroupNameDialog } from '../../components/GroupNameDialog';
import { DetectionBox } from '../../components/ImageModal';
import { Detection } from '../../types/electron';

interface ReidRun { id: number; name: string; species: string; created_at: number; individual_count: number; detection_count: number; }
interface ReidDetection { id: number; image_id: number; label: string; confidence: number; detection_confidence: number; x1: number; y1: number; x2: number; y2: number; source: string; batch_id: number; created_at: number; image_path: string; image_preview_path?: string; }
interface ReidIndividual { id: number; run_id: number; name: string; display_name: string; color: string; created_at: number; member_count: number; detections: ReidDetection[]; }

// Skeleton Card for loading state
const SkeletonCard: React.FC = () => {
    const theme = useTheme();
    return (
        <Box sx={{ borderRadius: 2, overflow: 'hidden', border: `1px solid ${theme.palette.divider}`, bgcolor: theme.palette.mode === 'light' ? '#F9F9F9' : theme.palette.background.paper }}>
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
        <Box onClick={onClick} sx={{ cursor: 'pointer', borderRadius: 2, overflow: 'hidden', transition: 'all 0.15s', border: `1px solid ${theme.palette.divider}`, bgcolor: theme.palette.mode === 'light' ? '#F9F9F9' : theme.palette.background.paper, '&:hover': { borderColor: individual.color } }}>
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

const IndividualModal: React.FC<{ open: boolean; onClose: () => void; individual: ReidIndividual | null; imageUrls: Map<string, string>; fullImageUrls: Map<string, string>; loadFullImage: (path: string) => void }> = ({ open, onClose, individual, imageUrls, fullImageUrls, loadFullImage }) => {
    const theme = useTheme();
    const [currentIndex, setCurrentIndex] = useState(0);
    const [zoom, setZoom] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const dragStart = useRef({ x: 0, y: 0 });
    const imageRef = useRef<HTMLImageElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [imgDims, setImgDims] = useState({ natural: { width: 0, height: 0 }, displayed: { width: 0, height: 0 } });

    const detections = individual?.detections || [];
    const currentDet = detections[currentIndex];
    const currentUrl = currentDet ? (fullImageUrls.get(currentDet.image_path) || imageUrls.get(currentDet.image_path)) : undefined;

    useEffect(() => { if (open && currentDet) { setZoom(1); setPosition({ x: 0, y: 0 }); loadFullImage(currentDet.image_path); } }, [open, currentIndex, currentDet]);
    useEffect(() => { setCurrentIndex(0); }, [individual?.id]);

    const handleImageLoad = () => {
        if (imageRef.current && containerRef.current) {
            const img = imageRef.current, rect = containerRef.current.getBoundingClientRect();
            const aspect = img.naturalWidth / img.naturalHeight, cAspect = rect.width / rect.height;
            const [dw, dh] = aspect > cAspect ? [rect.width, rect.width / aspect] : [rect.height * aspect, rect.height];
            setImgDims({ natural: { width: img.naturalWidth, height: img.naturalHeight }, displayed: { width: dw, height: dh } });
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
                                        <Box sx={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`, transition: isDragging ? 'none' : 'transform 0.1s' }}>
                                            <DetectionBox
                                                bbox={bbox}
                                                detection={{ ...currentDet, label: individual.display_name } as Detection}
                                                zoom={zoom}
                                                containerWidth={imgDims.displayed.width}
                                                containerHeight={imgDims.displayed.height}
                                                useLiquidGlass={true}
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
                                            <Typography variant="caption" color="text.secondary">{(det.confidence * 100).toFixed(0)}% confidence</Typography>
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

    const refreshData = useCallback(() => {
        setRefreshTrigger(prev => prev + 1);
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

    if (loading) return <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', pt: 8 }}><CircularProgress /></Box>;

    return (
        <Box sx={{ pt: '64px', px: 3, pb: 3, minHeight: '100vh' }}>
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
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 2 }}>
                        <Fingerprint size={28} weight="duotone" />
                        <Typography variant="h5" fontWeight={600}>Re-identification</Typography>
                        <Typography variant="body2" color="text.secondary">{runs.length} run{runs.length !== 1 ? 's' : ''}</Typography>
                    </Box>
                    {runs.map(run => {
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
            <IndividualModal open={modalOpen} onClose={() => setModalOpen(false)} individual={selectedIndividual} imageUrls={imageUrls} fullImageUrls={fullImageUrls} loadFullImage={loadFullImage} />
        </Box>
    );
};

export default ReIDPage;
