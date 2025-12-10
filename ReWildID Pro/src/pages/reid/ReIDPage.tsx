import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useOutletContext, useNavigate } from 'react-router-dom';
import {
    Box, Typography, IconButton, Menu, MenuItem,
    Chip, alpha, useTheme, Collapse, Skeleton, Button,
    Switch, Tooltip, Divider
} from '@mui/material';
import {
    ArrowLineUp, Fingerprint, DotsThreeVertical, PencilSimple, Trash, CaretDown, CaretRight,
    Images as ImagesIcon, Sparkle, Gear
} from '@phosphor-icons/react';
import { LiquidGlassButton } from '../../components/LiquidGlassButton';
import { GroupNameDialog } from '../../components/GroupNameDialog';
import { LibrarySearchBar } from '../../components/library/LibrarySearchBar';
import { RefreshNotification } from '../../components/RefreshNotification';
import { ReidIndividual, ReidRun } from './types';
import { useColorMode } from '../../features/theme/ThemeContext';
import {
    getReidPageCache,
    setReidPageCache,
    invalidateReidPageCache,
    updateReidPageCacheImageUrls
} from './reidPageCache';

// Skeleton Card for loading state
const SkeletonCard: React.FC<{ hasGradient?: boolean }> = ({ hasGradient }) => {
    const theme = useTheme();
    const getBgColor = () => {
        if (hasGradient) {
            return theme.palette.mode === 'dark' ? 'rgba(30, 30, 36, 0.75)' : 'rgba(247, 249, 251, 0.75)';
        }
        return theme.palette.mode === 'light' ? '#F7F9FB' : theme.palette.background.paper;
    };
    return (
        <Box sx={{
            borderRadius: 2,
            overflow: 'hidden',
            border: `1px solid ${theme.palette.divider}`,
            bgcolor: getBgColor(),
            backdropFilter: hasGradient ? 'blur(12px)' : 'none',
            WebkitBackdropFilter: hasGradient ? 'blur(12px)' : 'none',
        }}>
            <Skeleton variant="rectangular" width="100%" height={130} animation="wave" />
            <Box sx={{ p: 1.25 }}>
                <Skeleton variant="text" width="70%" height={20} animation="wave" />
                <Skeleton variant="text" width="50%" height={16} animation="wave" />
            </Box>
        </Box>
    );
};

const IndividualCard: React.FC<{ individual: ReidIndividual; onClick: () => void; imageUrls: Map<string, string>; showColors?: boolean; hasGradient?: boolean }> = ({ individual, onClick, imageUrls, showColors = false, hasGradient }) => {
    const theme = useTheme();
    const firstDet = individual.detections[0];
    const thumbUrl = firstDet ? imageUrls.get(firstDet.image_preview_path || firstDet.image_path) : undefined;
    const getBgColor = () => {
        if (hasGradient) {
            return theme.palette.mode === 'dark' ? 'rgba(30, 30, 36, 0.75)' : 'rgba(247, 249, 251, 0.75)';
        }
        return theme.palette.mode === 'light' ? '#F7F9FB' : theme.palette.background.paper;
    };
    return (
        <Box onClick={onClick} onDragStart={(e: React.DragEvent) => e.preventDefault()} sx={{
            cursor: 'pointer',
            borderRadius: 2,
            overflow: 'hidden',
            transition: 'all 0.15s',
            border: `1px solid ${theme.palette.divider}`,
            bgcolor: getBgColor(),
            backdropFilter: hasGradient ? 'blur(12px)' : 'none',
            WebkitBackdropFilter: hasGradient ? 'blur(12px)' : 'none',
            userSelect: 'none',
            '&:hover': { borderColor: showColors ? individual.color : theme.palette.primary.main }
        }}>
            <Box sx={{ width: '100%', height: 130, bgcolor: theme.palette.mode === 'light' ? '#f0f0f0' : '#0a0a0a', position: 'relative', overflow: 'hidden' }}>
                {thumbUrl ? <Box component="img" src={thumbUrl} draggable={false} sx={{ width: '100%', height: '100%', objectFit: 'cover', userSelect: 'none', WebkitUserDrag: 'none', pointerEvents: 'none' }} /> : <Box sx={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Fingerprint size={40} weight="thin" color={theme.palette.text.disabled} /></Box>}
                {showColors && <Box sx={{ position: 'absolute', top: 8, left: 8, width: 12, height: 12, borderRadius: '50%', bgcolor: individual.color, border: '2px solid white', boxShadow: '0 1px 3px rgba(0,0,0,0.3)' }} />}
                <Box sx={{ position: 'absolute', bottom: 6, right: 6, display: 'flex', alignItems: 'center', gap: 0.4, bgcolor: 'rgba(0,0,0,0.6)', color: 'white', px: 0.8, py: 0.3, borderRadius: 1, fontSize: '12px' }}>
                    <ImagesIcon size={14} />{individual.member_count}
                </Box>
            </Box>
            <Box sx={{ p: 1.25 }}>
                <Typography variant="body2" fontWeight={600} noWrap sx={{ color: showColors ? individual.color : 'text.primary' }}>{individual.display_name}</Typography>
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
    showColors: boolean;
    hasGradient: boolean;
}

const RunGroup: React.FC<RunGroupProps> = ({ run, individuals, imageUrls, onIndividualClick, onMenuOpen, hasMore, loadingMore, onLoadMore, showColors, hasGradient }) => {
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
                    {individuals.map((ind) => <IndividualCard key={ind.id} individual={ind} onClick={() => onIndividualClick(ind)} imageUrls={imageUrls} showColors={showColors} hasGradient={hasGradient} />)}
                    {loadingMore && Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={`skeleton-${i}`} hasGradient={hasGradient} />)}
                </Box>
                {hasMore && <Box ref={loadMoreRef} sx={{ height: 20, mt: 2 }} />}
            </Collapse>
        </Box>
    );
};

const PAGE_SIZE = 24;

const ReIDPage: React.FC = () => {
    const theme = useTheme();
    const { colorTheme } = useColorMode();
    const hasGradient = colorTheme.gradient !== 'none' || !!colorTheme.special || !!colorTheme.image;
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
    const [showColors, setShowColors] = useState(() => {
        const saved = localStorage.getItem('reid_showColors');
        return saved === 'true'; // default false
    });

    // Sync settings from Settings page
    useEffect(() => {
        const handleStorageChange = (e: StorageEvent) => {
            if (e.key === 'mediaExplorer_useLiquidGlass' && e.newValue !== null) {
                setUseLiquidGlass(e.newValue === 'true');
            } else if (e.key === 'mediaExplorer_useRayTracedGlass' && e.newValue !== null) {
                setUseRayTracedGlass(e.newValue === 'true');
            } else if (e.key === 'reid_showColors' && e.newValue !== null) {
                setShowColors(e.newValue === 'true');
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

    // Track paths that are currently being loaded to prevent duplicate requests
    const loadingPathsRef = useRef<Set<string>>(new Set());

    const loadImageByPath = useCallback(async (path: string) => {
        // Skip if already loaded or currently loading
        if (loadingPathsRef.current.has(path)) return;

        // Mark as loading
        loadingPathsRef.current.add(path);

        try {
            const response = await window.api.viewImage(path);
            if (response.ok && response.data) {
                const blob = new Blob([response.data as unknown as BlobPart], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                setImageUrls(prev => new Map(prev).set(path, url));
            }
        } catch (e) {
            console.error('Failed to load image:', path, e);
            // Remove from loading so it can be retried
            loadingPathsRef.current.delete(path);
        }
    }, []);

    const loadImagesForIndividuals = useCallback((inds: ReidIndividual[]) => {
        for (const ind of inds) {
            for (const det of ind.detections) {
                const path = det.image_preview_path || det.image_path;
                if (path) loadImageByPath(path);
            }
        }
    }, [loadImageByPath]);

    // Restore scroll position on mount
    useEffect(() => {
        const savedPosition = sessionStorage.getItem('reid_scroll_position');
        if (savedPosition) {
            // Short delay to ensure content is rendered (especially if data is loaded from cache or fast fetch)
            // If data loading takes time, we might need to restore after data load.
            // But runs are loaded in useEffect. 
            // We can just store the position and let the data loading effect trigger the scroll?
            // Or just try to scroll after a delay.
        }
    }, []);

    // Initial load - fetch runs and first page of each (or restore from cache)
    useEffect(() => {
        const loadData = async () => {
            // Check cache first (for back navigation)
            const cached = getReidPageCache();
            if (cached && refreshTrigger === 0) {
                // Restore from cache - instant!
                setRuns(cached.runs);
                setIndividuals(cached.individuals);
                setPagination(cached.pagination);
                setImageUrls(cached.imageUrls);
                setLoading(false);

                // Restore scroll position
                const savedPosition = sessionStorage.getItem('reid_scroll_position');
                if (savedPosition) {
                    setTimeout(() => {
                        const mainEl = document.querySelector('main');
                        if (mainEl) {
                            mainEl.scrollTop = parseInt(savedPosition, 10);
                            sessionStorage.removeItem('reid_scroll_position');
                        }
                    }, 50);
                }
                return;
            }

            // Cache miss or refresh triggered - fetch from API
            if (refreshTrigger > 0) {
                invalidateReidPageCache();
            }

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

                    // Save to cache for back navigation
                    // imageUrls will be populated asynchronously by loadImagesForIndividuals
                    setReidPageCache({
                        runs: sortedRuns,
                        individuals: newIndividuals,
                        pagination: newPagination,
                        imageUrls: new Map()
                    });

                    // Restore scroll position after data is loaded
                    const savedPosition = sessionStorage.getItem('reid_scroll_position');
                    if (savedPosition) {
                        setTimeout(() => {
                            const mainEl = document.querySelector('main');
                            if (mainEl) {
                                mainEl.scrollTop = parseInt(savedPosition, 10);
                                sessionStorage.removeItem('reid_scroll_position');
                            }
                        }, 100);
                    }
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

    // Sync imageUrls to cache as they are loaded asynchronously
    useEffect(() => {
        if (imageUrls.size > 0) {
            updateReidPageCacheImageUrls(imageUrls);
        }
    }, [imageUrls]);

    const handleMenuOpen = (e: React.MouseEvent<HTMLElement>, runId: number) => { setMenuAnchor(e.currentTarget); setSelectedRunId(runId); };
    const handleMenuClose = () => { setMenuAnchor(null); setSelectedRunId(null); };
    const handleRename = () => { const run = runs.find(r => r.id === selectedRunId); if (run) { setRunToRename({ id: run.id, name: run.name }); setRenameDialogOpen(true); } handleMenuClose(); };
    const handleConfirmRename = async (newName: string) => { if (runToRename) { await window.api.updateReidRunName(runToRename.id, newName); setRefreshTrigger(t => t + 1); } setRenameDialogOpen(false); setRunToRename(null); };
    const handleDelete = async () => { if (selectedRunId && window.confirm('Delete this ReID run?')) { await window.api.deleteReidRun(selectedRunId); setRefreshTrigger(t => t + 1); } handleMenuClose(); };
    const handleIndividualClick = (ind: ReidIndividual) => {
        const mainEl = document.querySelector('main');
        if (mainEl) {
            sessionStorage.setItem('reid_scroll_position', mainEl.scrollTop.toString());
        }
        navigate(`/reid/run/${ind.run_id}/individual/${ind.id}`, { state: { individual: ind } });
    };

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
    // Logic moved to routing (App.tsx + IndividualDetailView.tsx)

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
                                showColors={showColors}
                                hasGradient={hasGradient}
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

                <Divider sx={{ my: 1 }} />

                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.5 }}>
                    <Typography variant="body2">
                        Show ID Colors
                    </Typography>
                    <Switch
                        size="small"
                        checked={showColors}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                            setShowColors(e.target.checked);
                            localStorage.setItem('reid_showColors', e.target.checked.toString());
                        }}
                    />
                </Box>
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
