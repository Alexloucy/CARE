import React, { useMemo, useState, useEffect } from 'react';
import { useOutletContext } from 'react-router-dom';
import { Menu, MenuItem, useTheme } from '@mui/material';
import { ArrowLineUp, CheckSquare, Funnel, PencilSimple, Trash } from '@phosphor-icons/react';
import { Box, Tooltip } from '@mui/material';
import { LiquidGlassButton } from '../../components/LiquidGlassButton';
import { DBImage } from '../../types/electron';

// Hooks
import { useImageLoader } from '../../hooks/useImageLoader';
// import { useLibraryData } from '../../hooks/useLibraryData'; // Replaced with custom loading
import { useSelection } from '../../hooks/useSelection';

// Components
import { LibraryFilter } from '../../components/library/LibraryFilterDialog';
import { MediaExplorer } from '../../components/library/MediaExplorer';
import { GroupNameDialog } from '../../components/GroupNameDialog';
import { RefreshNotification } from '../../components/RefreshNotification';
import { DateSection, GroupData } from '../../types/library';

// Utils
import { triggerUpload } from '../../utils/navigationEvents';

const ClassificationPage: React.FC = () => {
    const theme = useTheme();
    const { leftSidebarOpen, rightSidebarOpen } = useOutletContext<{ leftSidebarOpen: boolean; rightSidebarOpen: boolean }>();

    // Scroll state for floating buttons
    const [isScrolled, setIsScrolled] = useState(false);

    // 1. Filter & Search State
    const [filterDialogOpen, setFilterDialogOpen] = useState(false);
    const [activeFilter, setActiveFilter] = useState<LibraryFilter | null>(null);
    const [searchQuery, setSearchQuery] = useState('');

    // 2. Data & Loading
    const [loading, setLoading] = useState(true); // Start true to show skeleton initially
    const [filteredDateSections, setFilteredDateSections] = useState<DateSection[]>([]);
    const [fullDateSections, setFullDateSections] = useState<DateSection[]>([]); // Needed for timeline?
    const [refreshTrigger, setRefreshTrigger] = useState(0);
    const [availableSpecies, setAvailableSpecies] = useState<string[]>([]);

    // Detection Batch (Group) Actions State
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    const [selectedBatchId, setSelectedBatchId] = useState<number | null>(null);
    const [renameDialogOpen, setRenameDialogOpen] = useState(false);
    const [batchToRename, setBatchToRename] = useState<{ id: number; name: string } | null>(null);

    const refreshLibrary = async () => {
        setRefreshTrigger(prev => prev + 1);
    };

    // Listen for refresh events from TaskPanel
    useEffect(() => {
        const handleRefresh = (e: CustomEvent<{ page: string }>) => {
            if (e.detail.page === 'classification') {
                refreshLibrary();
            }
        };
        window.addEventListener('trigger-refresh', handleRefresh as EventListener);
        return () => window.removeEventListener('trigger-refresh', handleRefresh as EventListener);
    }, []);

    // Data Loading Effect
    useEffect(() => {
        const loadData = async () => {
            setLoading(true);
            try {
                // Fetch available species
                const speciesRes = await window.api.getAvailableSpecies();
                console.log(speciesRes)
                if (speciesRes.ok && speciesRes.species) {
                    setAvailableSpecies(speciesRes.species);
                }

                const batchesRes = await window.api.getDetectionBatches();
                if (batchesRes.ok && batchesRes.batches) {
                    // Sort batches by date descending
                    const sortedBatches = batchesRes.batches.sort((a, b) => b.created_at - a.created_at);

                    const sectionsMap = new Map<string, GroupData[]>(); // DateKey -> Groups

                    for (const batch of sortedBatches) {
                        // Apply filters
                        const detRes = await window.api.getDetectionsForBatch(
                            batch.id,
                            activeFilter?.species,
                            activeFilter?.minConfidence
                        );
                        if (detRes.ok && detRes.detections) {
                            // Group detections by Image ID
                            const imagesMap = new Map<number, DBImage>();

                            for (const d of detRes.detections) {
                                // d is (Detection & Image) from backend query
                                const imageId = (d as any).id || d.image_id; // In my query I aliased images.id as 'id'

                                if (!imagesMap.has(imageId)) {
                                    imagesMap.set(imageId, {
                                        id: imageId,
                                        group_id: 0, // Placeholder
                                        original_path: (d as any).original_path || '',
                                        preview_path: (d as any).preview_path,
                                        date_added: (d as any).date_added,
                                        group_name: batch.name,
                                        group_created_at: batch.created_at,
                                        detections: []
                                    });
                                }

                                // Add detection info
                                imagesMap.get(imageId)?.detections?.push({
                                    id: (d as any).detection_id,
                                    label: d.label,
                                    confidence: d.confidence,
                                    detection_confidence: d.detection_confidence,
                                    x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2,
                                    source: d.source,
                                    created_at: (d as any).detection_created_at,
                                    batch_id: d.batch_id,
                                    image_id: d.image_id
                                });
                            }

                            const images = Array.from(imagesMap.values());

                            if (images.length > 0) {
                                // Group by Date
                                const dateObj = new Date(batch.created_at);
                                const y = dateObj.getFullYear();
                                const m = String(dateObj.getMonth() + 1).padStart(2, '0');
                                const d = String(dateObj.getDate()).padStart(2, '0');
                                const dateKey = `${y}${m}${d}`;

                                if (!sectionsMap.has(dateKey)) {
                                    sectionsMap.set(dateKey, []);
                                }
                                sectionsMap.get(dateKey)?.push({
                                    id: batch.id,
                                    name: batch.name,
                                    created_at: batch.created_at,
                                    images: images
                                });
                            }
                        }
                    }

                    // Convert map to array and sort
                    const sections: DateSection[] = Array.from(sectionsMap.entries())
                        .map(([date, groups]) => ({
                            date,
                            groups: groups.sort((a, b) => (b.created_at || 0) - (a.created_at || 0)) // Sort groups new to old
                        }))
                        .sort((a, b) => b.date.localeCompare(a.date));

                    setFullDateSections(sections);
                    setFilteredDateSections(sections);
                }
            } catch (e) {
                console.error("Failed to load detections:", e);
            } finally {
                setLoading(false);
            }
        };
        loadData();
    }, [refreshTrigger, activeFilter]); // Re-fetch when filter changes

    // Search Filtering Effect
    useEffect(() => {
        if (!searchQuery.trim()) {
            setFilteredDateSections(fullDateSections);
            return;
        }

        const query = searchQuery.toLowerCase();
        const filtered = fullDateSections.map(section => {
            const filteredGroups = section.groups.map(group => {
                // Filter images by label or path
                const filteredImages = group.images.filter(img => {
                    const label = img.detections?.[0]?.label?.toLowerCase() || '';
                    const path = img.original_path.toLowerCase();
                    return label.includes(query) || path.includes(query);
                });

                // Also match group name
                if (group.name.toLowerCase().includes(query)) {
                    return group; // Keep all images if group name matches
                }

                return {
                    ...group,
                    images: filteredImages
                };
            }).filter(group => group.images.length > 0);

            return {
                ...section,
                groups: filteredGroups
            };
        }).filter(section => section.groups.length > 0);

        setFilteredDateSections(filtered);
    }, [searchQuery, fullDateSections]);

    // 3. Image Loading
    const { imageUrls, fullImageUrls, loadImage, loadFullImage } = useImageLoader();

    // 4. Selection
    const {
        isSelectionMode,
        selectedIds: selectedImageIds,
        toggleSelectionMode,
        toggleItem: toggleImageSelection,
        clearSelection,
        setIsSelectionMode,
        setSelection
    } = useSelection<number>();

    // Derived State
    const allImages = useMemo(() => {
        return filteredDateSections.flatMap(section => section.groups.flatMap(group => group.images));
    }, [filteredDateSections]);

    // Batch Actions
    const handleBatchDelete = async () => {
        if (selectedImageIds.size === 0) return;

        // Gather all detections from selected images
        const detectionsToDelete = allImages
            .filter(img => selectedImageIds.has(img.id))
            .flatMap(img => img.detections || []);

        if (detectionsToDelete.length === 0) {
            alert('No detections found in the selected images.');
            return;
        }

        if (window.confirm(`Are you sure you want to delete ${detectionsToDelete.length} detections from ${selectedImageIds.size} images?`)) {
            try {
                setLoading(true);
                for (const det of detectionsToDelete) {
                    await window.api.deleteDetection(det.id);
                }
                await refreshLibrary();
                clearSelection();
            } catch (error) {
                console.error('Batch delete error:', error);
                alert('Failed to delete some detections.');
            } finally {
                setLoading(false);
            }
        }
    };

    const handleBatchDetect = async () => {
        if (selectedImageIds.size === 0) return;
        const paths: string[] = [];
        const ids: number[] = [];
        allImages.forEach(img => {
            if (selectedImageIds.has(img.id)) {
                paths.push(img.original_path);
                ids.push(img.id);
            }
        });
        if (paths.length === 0) return;

        try {
            await window.api.detect(paths, (txt) => console.log(txt), ids);
            setIsSelectionMode(false);
            clearSelection();
        } catch (error) {
            console.error('Batch detect error:', error);
            alert('Failed to start detection: ' + error);
        }
    };

    const handleBatchSave = async () => {
        if (selectedImageIds.size === 0) return;
        const paths: string[] = [];
        allImages.forEach(img => {
            if (selectedImageIds.has(img.id)) paths.push(img.original_path);
        });
        if (paths.length === 0) return;

        try {
            const result = await window.api.saveImages(paths);
            if (result.ok) {
                alert(`Successfully saved ${result.successCount} images.`);
                clearSelection();
            }
        } catch (e) {
            console.error(e);
        }
    };

    const handleDeleteImage = async (image: DBImage) => {
        await window.api.deleteImage(image.id);
        await refreshLibrary();
    };

    // ReID Handler
    const handleReID = async (images: DBImage[], species: string) => {
        const imageIds = images.map(img => img.id);
        if (imageIds.length === 0) {
            alert('No images selected for ReID.');
            return;
        }

        try {
            const result = await window.api.smartReID(imageIds, species);

            if (result.ok) {
                clearSelection();
            } else {
                alert('ReID failed: ' + result.error);
            }
        } catch (error) {
            console.error('ReID error:', error);
            alert('ReID failed: ' + error);
        }
    };

    // Detection Batch (Group) Menu Handlers
    const handleMenuOpen = (e: React.MouseEvent<HTMLElement>, groupId: number) => {
        e.stopPropagation();
        setAnchorEl(e.currentTarget);
        setSelectedBatchId(groupId);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
        setSelectedBatchId(null);
    };

    const handleRenameBatchClick = () => {
        if (selectedBatchId === null) return;
        const batch = fullDateSections
            .flatMap(s => s.groups)
            .find(g => g.id === selectedBatchId);
        if (batch) {
            setBatchToRename({ id: batch.id, name: batch.name });
            setRenameDialogOpen(true);
        }
        handleMenuClose();
    };

    const handleConfirmRename = async (newName: string) => {
        if (!batchToRename) return;
        try {
            await window.api.updateDetectionBatchName(batchToRename.id, newName);
            await refreshLibrary();
        } catch (error) {
            console.error('Rename batch error:', error);
            alert('Failed to rename detection batch.');
        }
        setRenameDialogOpen(false);
        setBatchToRename(null);
    };

    const handleDeleteBatch = async () => {
        if (selectedBatchId === null) return;
        const batch = fullDateSections
            .flatMap(s => s.groups)
            .find(g => g.id === selectedBatchId);
        if (!batch) return;

        if (window.confirm(`Are you sure you want to delete the detection batch "${batch.name}"? This will delete all detections in this batch.`)) {
            try {
                await window.api.deleteDetectionBatch(selectedBatchId);
                await refreshLibrary();
            } catch (error) {
                console.error('Delete batch error:', error);
                alert('Failed to delete detection batch.');
            }
        }
        handleMenuClose();
    };

    return (
        <>
            <RefreshNotification
                watchJobTypes={['detect']}
                onRefresh={refreshLibrary}
                message="Classification completed"
            />
            <MediaExplorer
                title="Classification"
                loading={loading}
                dateSections={filteredDateSections}
                fullDateSections={fullDateSections}
                imageUrls={imageUrls}
                fullImageUrls={fullImageUrls}
                allImages={allImages}
                loadImage={loadImage}
                loadFullImage={loadFullImage}
                activeFilter={activeFilter}
                onFilterChange={setActiveFilter}
                searchQuery={searchQuery}
                onSearchChange={setSearchQuery}
                filterDialogOpen={filterDialogOpen}
                setFilterDialogOpen={setFilterDialogOpen}
                isSelectionMode={isSelectionMode}
                selectedImageIds={selectedImageIds}
                toggleSelectionMode={toggleSelectionMode}
                toggleImageSelection={toggleImageSelection}
                setSelection={setSelection}
                clearSelection={clearSelection}
                setIsSelectionMode={setIsSelectionMode}
                onBatchDelete={handleBatchDelete}
                onBatchDetect={handleBatchDetect}
                onBatchSave={handleBatchSave}
                onDeleteImage={handleDeleteImage}
                leftSidebarOpen={leftSidebarOpen}
                rightSidebarOpen={rightSidebarOpen}
                availableSpecies={availableSpecies}
                onGroupMenuOpen={handleMenuOpen}
                aiButtonMode="reid"
                onReID={handleReID}
                onUpload={triggerUpload}
                onScrollStateChange={setIsScrolled}
                groupMenu={
                    <Menu
                        anchorEl={anchorEl}
                        open={Boolean(anchorEl)}
                        onClose={handleMenuClose}
                        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
                        PaperProps={{
                            elevation: 0,
                            sx: {
                                backgroundColor: theme.palette.mode === 'light' ? 'rgba(255, 255, 255, 0.85)' : 'rgba(45, 45, 45, 0.85)',
                                backdropFilter: 'blur(8px)',
                                borderRadius: '8px',
                                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
                                border: `1px solid ${theme.palette.divider}`,
                                minWidth: '160px',
                                mt: 0.5
                            }
                        }}
                        MenuListProps={{ sx: { padding: '6px' } }}
                    >
                        <MenuItem onClick={handleRenameBatchClick} sx={{ borderRadius: '6px', margin: '2px 0', gap: 1 }}>
                            <PencilSimple size={18} /> Rename
                        </MenuItem>
                        <MenuItem onClick={handleDeleteBatch} sx={{ borderRadius: '6px', margin: '2px 0', gap: 1, color: 'error.main' }}>
                            <Trash size={18} /> Delete
                        </MenuItem>
                    </Menu>
                }
            />

            <GroupNameDialog
                open={renameDialogOpen}
                onClose={() => { setRenameDialogOpen(false); setBatchToRename(null); }}
                onConfirm={handleConfirmRename}
                title="Rename Classification Batch"
                initialValue={batchToRename?.name || ''}
            />

            {/* Floating Action Buttons - Show when scrolled */}
            <Box
                sx={{
                    position: 'fixed',
                    top: 80,
                    right: rightSidebarOpen ? 228 : 16,
                    display: 'flex',
                    flexDirection: 'row',
                    gap: 1,
                    zIndex: 1000,
                    p: 1.5,
                    opacity: isScrolled ? 1 : 0,
                    transform: isScrolled ? 'translateX(0)' : 'translateX(calc(100% + 32px))',
                    transition: 'transform 0.5s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.3s ease, right 0.3s ease',
                    pointerEvents: isScrolled ? 'auto' : 'none'
                }}
            >
                <Tooltip title="Filter">
                    <span>
                        <LiquidGlassButton
                            size={32}
                            icon={<Funnel size={16} weight={activeFilter ? 'fill' : 'regular'} />}
                            onClick={() => setFilterDialogOpen(true)}
                        />
                    </span>
                </Tooltip>
                <Tooltip title={isSelectionMode ? 'Exit Selection' : 'Select'}>
                    <span>
                        <LiquidGlassButton
                            size={32}
                            icon={<CheckSquare size={16} weight={isSelectionMode ? 'fill' : 'regular'} />}
                            onClick={toggleSelectionMode}
                        />
                    </span>
                </Tooltip>
                <Tooltip title="Back to Top">
                    <span>
                        <LiquidGlassButton
                            size={32}
                            icon={<ArrowLineUp size={16} />}
                            onClick={() => {
                                const virtuosoContainer = document.querySelector('[data-virtuoso-scroller]');
                                if (virtuosoContainer) {
                                    virtuosoContainer.scrollTo({ top: 0, behavior: 'smooth' });
                                }
                            }}
                        />
                    </span>
                </Tooltip>
            </Box>
        </>
    );
};

export default ClassificationPage;
