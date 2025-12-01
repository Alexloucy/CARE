import React, { useEffect, useState, useMemo, useRef } from 'react';
import { Box, Typography, useTheme, Skeleton, Fade, IconButton, Menu, MenuItem, Divider, Button, Tooltip } from '@mui/material';
import { DBImage, FileDetails } from '../../types/electron';
import ImageModal from '../../components/ImageModal';
import ImageCard from '../../components/ImageCard';
import { UploadIcon, UploadSimple, DotsThreeVertical, Trash, PencilSimple, CheckSquare, DownloadSimple, X, Plus } from '@phosphor-icons/react';
import { GroupNameDialog } from '../../components/GroupNameDialog';

interface GroupData {
    id: number;
    name: string;
    created_at: number;
    images: DBImage[];
}

interface DateSection {
    date: string; // YYYYMMDD
    groups: GroupData[];
}

const LibraryPage: React.FC = () => {
    const [loading, setLoading] = useState(true);
    const [dateSections, setDateSections] = useState<DateSection[]>([]);
    const [imageUrls, setImageUrls] = useState<Record<number, string>>({}); // Map ID -> URL (Thumbnails)
    const [fullImageUrls, setFullImageUrls] = useState<Record<number, string>>({}); // Map ID -> URL (Full Res)
    const [isDragging, setIsDragging] = useState(false);
    const theme = useTheme();

    // Selection State
    const [isSelectionMode, setIsSelectionMode] = useState(false);
    const [selectedImageIds, setSelectedImageIds] = useState<Set<number>>(new Set());

    // Modal State
    const [selectedImage, setSelectedImage] = useState<{ image: DBImage, url: string } | null>(null);

    // Group Name Dialog State
    const [groupNameDialogOpen, setGroupNameDialogOpen] = useState(false);
    const [pendingUploadFiles, setPendingUploadFiles] = useState<string[]>([]);

    // Rename Group Dialog State
    const [renameDialogOpen, setRenameDialogOpen] = useState(false);
    const [groupToRename, setGroupToRename] = useState<{ id: number, name: string } | null>(null);

    // Menu State
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    const [menuGroupId, setMenuGroupId] = useState<number | null>(null);

    const fetchLibrary = async () => {
        try {
            setLoading(true);
            const response = await window.api.getImages();

            if (!response.ok || !response.images) {
                console.error('Failed to fetch library:', response.error);
                setLoading(false);
                return;
            }

            const images = response.images;
            const groupsMap: Record<number, GroupData> = {};

            // Group by Group ID first
            images.forEach(img => {
                if (!groupsMap[img.group_id]) {
                    groupsMap[img.group_id] = {
                        id: img.group_id,
                        name: img.group_name,
                        created_at: img.group_created_at,
                        images: []
                    };
                }
                groupsMap[img.group_id].images.push(img);
            });

            // Group by Date (using group_created_at)
            const dateMap: Record<string, GroupData[]> = {};

            Object.values(groupsMap).forEach(group => {
                const date = new Date(group.created_at).toISOString().split('T')[0].replace(/-/g, ''); // YYYYMMDD
                if (!dateMap[date]) {
                    dateMap[date] = [];
                }
                dateMap[date].push(group);
            });

            // Sort Dates DESC, Groups DESC (by created_at)
            const sortedDates = Object.keys(dateMap).sort((a, b) => b.localeCompare(a));
            const newDateSections: DateSection[] = sortedDates.map(date => ({
                date,
                groups: dateMap[date].sort((a, b) => b.created_at - a.created_at)
            }));

            setDateSections(newDateSections);
        } catch (error) {
            console.error('Error loading library:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchLibrary();
    }, []);

    const formatDate = (dateStr: string) => {
        if (dateStr.length !== 8) return dateStr;
        const year = dateStr.substring(0, 4);
        const month = dateStr.substring(4, 6);
        const day = dateStr.substring(6, 8);
        const date = new Date(parseInt(year), parseInt(month) - 1, parseInt(day));
        return date.toLocaleDateString(undefined, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    };

    const loadFullImage = async (image: DBImage) => {
        if (fullImageUrls[image.id]) return;

        try {
            // Always load original path for full view
            const response = await window.api.viewImage(image.original_path);
            
            if (response.ok && response.data) {
                const blob = new Blob([response.data as unknown as BlobPart], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                setFullImageUrls(prev => ({ ...prev, [image.id]: url }));
            }
        } catch (error) {
            console.error(`Failed to load full image ${image.original_path}:`, error);
        }
    };

    const loadImage = async (image: DBImage) => {
        if (imageUrls[image.id]) return;

        try {
            // Prioritize preview path if available
            const pathToCheck = image.preview_path || image.original_path;
            const response = await window.api.viewImage(pathToCheck);
            
            if (response.ok && response.data) {
                const blob = new Blob([response.data as unknown as BlobPart], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                setImageUrls(prev => ({ ...prev, [image.id]: url }));
            }
        } catch (error) {
            console.error(`Failed to load image ${image.original_path}:`, error);
        }
    };

    // Load full image when selected
    useEffect(() => {
        if (selectedImage) {
            loadFullImage(selectedImage.image);
        }
    }, [selectedImage?.image.id]);

    // Update selectedImage URL when full image loads
    useEffect(() => {
        if (selectedImage) {
            if (fullImageUrls[selectedImage.image.id]) {
                setSelectedImage(prev => prev ? { ...prev, url: fullImageUrls[selectedImage.image.id] } : null);
            }
        }
    }, [fullImageUrls, selectedImage?.image.id]);

    // Listen for job completions to refresh library
    const seenJobIds = useRef<Set<string>>(new Set());

    useEffect(() => {
        const removeListener = window.api.onJobUpdate((jobs) => {
            let shouldRefresh = false;
            jobs.forEach(job => {
                if (job.type === 'import' && job.status === 'completed') {
                    if (!seenJobIds.current.has(job.id)) {
                        seenJobIds.current.add(job.id);
                        shouldRefresh = true;
                    }
                }
            });
            
            if (shouldRefresh) {
                fetchLibrary();
            }
        });
        return removeListener;
    }, []);

    // Flatten images for navigation
    const allImages = useMemo(() => {
        return dateSections.flatMap(section => section.groups.flatMap(group => group.images));
    }, [dateSections]);

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

    // Drag & Drop Handlers
    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        if (e.currentTarget.contains(e.relatedTarget as Node)) {
            return;
        }
        setIsDragging(false);
    };

    const handleDrop = async (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        const files = Array.from(e.dataTransfer.files);
        if (files.length === 0) return;

        const paths = files.map(file => window.api.getPathForFile(file));
        processUploadPaths(paths);
    };

    const handleUploadClick = async () => {
        const result = await window.api.openFileDialog();
        if (!result.canceled && result.filePaths.length > 0) {
            processUploadPaths(result.filePaths);
        }
    };

    const processUploadPaths = async (paths: string[]) => {
        // Check if all paths are directories
        const areAllDirectories = await Promise.all(paths.map(path => window.api.checkIsDirectory(path)));
        const allDirs = areAllDirectories.every(isDir => isDir);

        if (allDirs) {
            // If all are directories, upload directly (backend handles group creation from folder name)
            // Async upload
            try {
                const result = await window.api.uploadPaths(paths);
                if (result.ok) {
                   // Import started
                } else {
                    console.error('Upload failed:', result.error);
                    alert('Upload failed: ' + result.error);
                }
            } catch (error) {
                console.error('Upload error:', error);
                alert('Upload error occurred.');
            }
        } else {
            // If any are files, prompt for group name
            setPendingUploadFiles(paths);
            setGroupNameDialogOpen(true);
        }
    };

    const handleConfirmUpload = async (name: string) => {
        setGroupNameDialogOpen(false);
        try {
            const response = await window.api.uploadPaths(pendingUploadFiles, name);
            if (response.ok) {
                // Import started
            } else {
                console.error('Upload failed:', response.error);
            }
        } catch (error) {
            console.error('Error uploading:', error);
        } finally {
            setPendingUploadFiles([]);
        }
    };

    // Group Actions
    const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, groupId: number) => {
        setAnchorEl(event.currentTarget);
        setMenuGroupId(groupId);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
        setMenuGroupId(null);
    };

    const handleDeleteGroup = async () => {
        if (menuGroupId === null) return;
        if (window.confirm('Are you sure you want to delete this group and all its images?')) {
            await window.api.deleteGroup(menuGroupId);
            await fetchLibrary();
        }
        handleMenuClose();
    };

    const handleRenameGroupClick = () => {
        if (menuGroupId === null) return;
        // Find group name
        let groupName = '';
        for (const section of dateSections) {
            const group = section.groups.find(g => g.id === menuGroupId);
            if (group) {
                groupName = group.name;
                break;
            }
        }
        setGroupToRename({ id: menuGroupId, name: groupName });
        setRenameDialogOpen(true);
        handleMenuClose();
    };

    const handleConfirmRename = async (newName: string) => {
        if (groupToRename) {
            await window.api.updateGroupName(groupToRename.id, newName);
            await fetchLibrary();
        }
        setRenameDialogOpen(false);
        setGroupToRename(null);
    };

    const handleDeleteImage = async () => {
        if (!selectedImage) return;
        // Confirm is handled in Modal
        await window.api.deleteImage(selectedImage.image.id);
        setSelectedImage(null);
        await fetchLibrary();
    };

    // Selection Logic
    const toggleSelectionMode = () => {
        setIsSelectionMode(!isSelectionMode);
        setSelectedImageIds(new Set());
    };

    const toggleImageSelection = (id: number) => {
        const newSet = new Set(selectedImageIds);
        if (newSet.has(id)) {
            newSet.delete(id);
        } else {
            newSet.add(id);
        }
        setSelectedImageIds(newSet);
    };

    const handleBatchDelete = async () => {
        if (selectedImageIds.size === 0) return;
        if (window.confirm(`Are you sure you want to delete ${selectedImageIds.size} images?`)) {
            setLoading(true);
            try {
                for (const id of selectedImageIds) {
                    await window.api.deleteImage(id);
                }
                await fetchLibrary();
                setSelectedImageIds(new Set());
                setIsSelectionMode(false);
            } catch (error) {
                console.error('Batch delete error:', error);
            } finally {
                setLoading(false);
            }
        }
    };

    const handleBatchSave = async () => {
        if (selectedImageIds.size === 0) return;

        // Collect paths
        const paths: string[] = [];
        allImages.forEach(img => {
            if (selectedImageIds.has(img.id)) {
                paths.push(img.original_path);
            }
        });

        if (paths.length === 0) return;

        setLoading(true);
        try {
            const result = await window.api.saveImages(paths);
            if (result.ok) {
                alert(`Successfully saved ${result.successCount} images.`);
                setSelectedImageIds(new Set());
                setIsSelectionMode(false);
            } else if (result.error !== 'Operation canceled') {
                alert(`Save failed: ${result.error}`);
            }
        } catch (error) {
            console.error('Batch save error:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Box
            sx={{
                height: '100%',
                position: 'relative',
                outline: 'none',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column'
            }}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
            {/* Drag Overlay */}
            <Fade in={isDragging}>
                <Box sx={{
                    position: 'fixed',
                    inset: 0,
                    zIndex: 9999,
                    bgcolor: theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.8)',
                    backdropFilter: 'blur(8px)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'all 0.2s ease',
                    pointerEvents: 'none'
                }}>
                    <UploadIcon size={80} color={theme.palette.primary.main} weight="regular" />
                    <Typography variant="h3" sx={{ mt: 4, fontWeight: 400, color: theme.palette.text.primary }}>
                        Drop to Upload
                    </Typography>
                </Box>
            </Fade>

            {/* Header */}
            <Box sx={{
                p: 3,
                px: 4,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                // Removed borderBottom for cleaner look
                bgcolor: theme.palette.background.default,
                zIndex: 10
            }}>
                <Typography variant="h4" fontWeight="bold">Library</Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                    <Tooltip title={isSelectionMode ? "Cancel Selection" : "Select Items"}>
                        <IconButton
                            onClick={toggleSelectionMode}
                            color={isSelectionMode ? "primary" : "default"}
                            sx={{
                                bgcolor: isSelectionMode ? (theme.palette.mode === 'light' ? 'rgba(25, 118, 210, 0.08)' : 'rgba(144, 202, 249, 0.16)') : 'transparent',
                                '&:hover': {
                                    bgcolor: isSelectionMode ? (theme.palette.mode === 'light' ? 'rgba(25, 118, 210, 0.12)' : 'rgba(144, 202, 249, 0.24)') : theme.palette.action.hover
                                }
                            }}
                        >
                            {isSelectionMode ? <X weight="bold" /> : <CheckSquare weight="regular" />}
                        </IconButton>
                    </Tooltip>
                    <Button
                        variant="contained"
                        startIcon={<Plus />}
                        onClick={handleUploadClick}
                        sx={{ borderRadius: 2, textTransform: 'none', px: 3 }}
                    >
                        Upload
                    </Button>
                </Box>
            </Box>

            {/* Content */}
            <Box sx={{ flex: 1, overflowY: 'auto', p: 4, pt: 0 }}>
                {
                    loading ? (
                        <Box sx={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                            gap: 2,
                            mt: 2
                        }}>
                            {[...Array(12)].map((_, i) => (
                                <Skeleton key={i} variant="rectangular" sx={{ borderRadius: 3, aspectRatio: '1/1', height: 'auto' }} />
                            ))}
                        </Box>
                    ) : (
                        <Box>
                            {dateSections.length === 0 ? (
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
                            ) : (
                                dateSections.map((section) => (
                                    <Box key={section.date} sx={{ mb: 5, mt: 2 }}>
                                        <Typography variant="h6" sx={{ mb: 2, fontWeight: 700, color: theme.palette.text.secondary, textTransform: 'uppercase', letterSpacing: '0.5px', fontSize: '0.875rem' }}>
                                            {formatDate(section.date)}
                                        </Typography>

                                        {section.groups.map(group => (
                                            <Box key={group.id} sx={{ mb: 4 }}>
                                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                                            {group.name}
                                                        </Typography>
                                                        <Typography variant="caption" color="text.secondary" sx={{ bgcolor: theme.palette.action.selected, px: 1, py: 0.5, borderRadius: 1 }}>
                                                            {group.images.length}
                                                        </Typography>
                                                    </Box>
                                                    <IconButton
                                                        size="small"
                                                        onClick={(e: React.MouseEvent<HTMLButtonElement>) => handleMenuOpen(e, group.id)}
                                                        sx={{ opacity: 0.6, '&:hover': { opacity: 1 } }}
                                                    >
                                                        <DotsThreeVertical size={20} />
                                                    </IconButton>
                                                </Box>

                                                <Box sx={{
                                                    display: 'grid',
                                                    gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                                                    gap: 2
                                                }}>
                                                    {group.images.map((img) => {
                                                        const fileDetails: FileDetails = {
                                                            name: img.original_path.split(/[\\/]/).pop() || 'image.jpg',
                                                            path: img.original_path,
                                                            isDirectory: false
                                                        };

                                                        return (
                                                            <Box key={img.id}>
                                                                <ImageCard
                                                                    file={fileDetails}
                                                                    date={section.date}
                                                                    loadImage={() => loadImage(img)}
                                                                    imageUrl={imageUrls[img.id]}
                                                                    onClick={() => {
                                                                        if (isSelectionMode) {
                                                                            toggleImageSelection(img.id);
                                                                        } else if (imageUrls[img.id]) {
                                                                            setSelectedImage({ image: img, url: imageUrls[img.id] });
                                                                        }
                                                                    }}
                                                                    selectable={isSelectionMode}
                                                                    selected={selectedImageIds.has(img.id)}
                                                                    onToggleSelection={() => toggleImageSelection(img.id)}
                                                                />
                                                            </Box>
                                                        );
                                                    })}
                                                </Box>
                                            </Box>
                                        ))}
                                    </Box>
                                ))
                            )}
                        </Box>
                    )
                }
            </Box>

            {/* Selection Toolbar */}
            <Fade in={isSelectionMode && selectedImageIds.size > 0}>
                <Box sx={{
                    position: 'absolute',
                    bottom: 32,
                    left: '50%',
                    transform: 'translateX(-50%)',
                    bgcolor: theme.palette.mode === 'dark' ? 'rgba(30,30,30,0.9)' : 'rgba(255,255,255,0.9)',
                    backdropFilter: 'blur(12px)',
                    borderRadius: 4,
                    boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
                    p: 1.5,
                    px: 3,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 3,
                    zIndex: 100,
                    border: `1px solid ${theme.palette.divider}`
                }}>
                    <Typography variant="subtitle2" fontWeight="600">
                        {selectedImageIds.size} Selected
                    </Typography>
                    <Divider orientation="vertical" flexItem sx={{ height: 20, my: 'auto' }} />
                    <Tooltip title="Save Selected">
                        <IconButton onClick={handleBatchSave} color="primary" size="small">
                            <DownloadSimple weight="bold" size={20} />
                        </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete Selected">
                        <IconButton onClick={handleBatchDelete} color="error" size="small">
                            <Trash weight="bold" size={20} />
                        </IconButton>
                    </Tooltip>
                </Box>
            </Fade>

            {/* Image Modal */}
            <ImageModal
                open={!!selectedImage}
                onClose={() => setSelectedImage(null)}
                imageUrl={selectedImage?.url}
                file={selectedImage ? {
                    name: selectedImage.image.original_path.split(/[\\/]/).pop() || 'image.jpg',
                    path: selectedImage.image.original_path,
                    isDirectory: false
                } : undefined}
                onNext={handleNext}
                onPrev={handlePrev}
                hasNext={selectedImage ? allImages.findIndex(img => img.id === selectedImage.image.id) < allImages.length - 1 : false}
                hasPrev={selectedImage ? allImages.findIndex(img => img.id === selectedImage.image.id) > 0 : false}
                onDelete={handleDeleteImage}
            />

            {/* Group Name Dialog (Upload) */}
            <GroupNameDialog
                open={groupNameDialogOpen}
                onClose={() => {
                    setGroupNameDialogOpen(false);
                    setPendingUploadFiles([]);
                }}
                onConfirm={handleConfirmUpload}
                title="Create New Group"
            />

            {/* Rename Group Dialog */}
            <GroupNameDialog
                open={renameDialogOpen}
                onClose={() => {
                    setRenameDialogOpen(false);
                    setGroupToRename(null);
                }}
                onConfirm={handleConfirmRename}
                title="Rename Group"
                initialValue={groupToRename?.name || ''}
            />

            {/* Group Action Menu */}
            <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
                anchorOrigin={{
                    vertical: 'bottom',
                    horizontal: 'right',
                }}
                transformOrigin={{
                    vertical: 'top',
                    horizontal: 'right',
                }}
                PaperProps={{
                    elevation: 0,
                    sx: {
                        backgroundColor: theme.palette.mode === 'light'
                            ? 'rgba(255, 255, 255, 0.85)'
                            : 'rgba(45, 45, 45, 0.85)',
                        backdropFilter: 'blur(8px)',
                        borderRadius: '8px',
                        boxShadow: theme.palette.mode === 'light'
                            ? '0 4px 20px rgba(0, 0, 0, 0.08)'
                            : '0 4px 20px rgba(0, 0, 0, 0.25)',
                        border: theme.palette.mode === 'light'
                            ? '1px solid rgba(230, 230, 230, 0.85)'
                            : '1px solid rgba(70, 70, 70, 0.85)',
                        minWidth: '160px',
                        mt: 0.5
                    }
                }}
                MenuListProps={{
                    sx: {
                        padding: '6px',
                    }
                }}
            >
                <MenuItem
                    onClick={handleRenameGroupClick}
                    sx={{
                        borderRadius: '6px',
                        margin: '2px 0',
                        gap: 1,
                        fontSize: '0.9rem',
                        '&:hover': {
                            backgroundColor: theme.palette.mode === 'light'
                                ? 'rgba(0, 0, 0, 0.04)'
                                : 'rgba(255, 255, 255, 0.08)'
                        }
                    }}
                >
                    <PencilSimple size={18} />
                    Rename
                </MenuItem>
                <MenuItem
                    onClick={handleDeleteGroup}
                    sx={{
                        borderRadius: '6px',
                        margin: '2px 0',
                        gap: 1,
                        fontSize: '0.9rem',
                        color: 'error.main',
                        '&:hover': {
                            backgroundColor: theme.palette.mode === 'light'
                                ? 'rgba(211, 47, 47, 0.08)'
                                : 'rgba(244, 67, 54, 0.12)'
                        }
                    }}
                >
                    <Trash size={18} />
                    Delete
                </MenuItem>
            </Menu>

        </Box >
    );
};

export default LibraryPage;
