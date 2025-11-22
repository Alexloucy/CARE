import React, { useEffect, useState, useMemo } from 'react';
import { Box, Typography, useTheme, Skeleton, Fade, IconButton, Menu, MenuItem, Divider } from '@mui/material';
import { DBImage, FileDetails } from '../../types/electron';
import ImageModal from '../../components/ImageModal';
import ImageCard from '../../components/ImageCard';
import { UploadIcon, UploadSimple, DotsThreeVertical, Trash, PencilSimple } from '@phosphor-icons/react';
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
    const [imageUrls, setImageUrls] = useState<Record<number, string>>({}); // Map ID -> URL
    const [isDragging, setIsDragging] = useState(false);
    const theme = useTheme();

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

    const loadImage = async (image: DBImage) => {
        if (imageUrls[image.id]) return;

        try {
            const response = await window.api.viewImage(image.original_path);
            if (response.ok && response.data) {
                const blob = new Blob([response.data as unknown as BlobPart], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                setImageUrls(prev => ({ ...prev, [image.id]: url }));
            }
        } catch (error) {
            console.error(`Failed to load image ${image.original_path}:`, error);
        }
    };

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
            setSelectedImage({ image: prevImage, url: imageUrls[prevImage.id] || '' });
        }
    };

    // Update selectedImage URL when it loads
    useEffect(() => {
        if (selectedImage && !selectedImage.url) {
            if (imageUrls[selectedImage.image.id]) {
                setSelectedImage(prev => prev ? { ...prev, url: imageUrls[selectedImage.image.id] } : null);
            }
        }
    }, [imageUrls, selectedImage]);

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

        // Check if all paths are directories
        const areAllDirectories = await Promise.all(paths.map(path => window.api.checkIsDirectory(path)));
        const allDirs = areAllDirectories.every(isDir => isDir);

        if (allDirs) {
            // If all are directories, upload directly (backend handles group creation from folder name)
            setLoading(true);
            try {
                const result = await window.api.uploadPaths(paths);
                if (result.ok) {
                    await fetchLibrary();
                } else {
                    console.error('Upload failed:', result.error);
                    alert('Upload failed: ' + result.error);
                }
            } catch (error) {
                console.error('Upload error:', error);
                alert('Upload error occurred.');
            } finally {
                setLoading(false);
            }
        } else {
            // If any are files, prompt for group name
            setPendingUploadFiles(paths);
            setGroupNameDialogOpen(true);
        }
    };

    const handleConfirmUpload = async (name: string) => {
        setGroupNameDialogOpen(false);
        setLoading(true);
        try {
            const response = await window.api.uploadPaths(pendingUploadFiles, name);
            if (response.ok) {
                console.log(`Successfully uploaded ${response.count} images.`);
                await fetchLibrary();
            } else {
                console.error('Upload failed:', response.error);
            }
        } catch (error) {
            console.error('Error uploading:', error);
        } finally {
            setLoading(false);
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

    return (
        <Box
            sx={{
                height: '100%',
                position: 'relative',
                outline: 'none',
                overflow: 'hidden'
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

            <Box sx={{ p: 4, height: '100%', overflowY: 'auto' }}>
                <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="h4" fontWeight="bold">Library</Typography>
                </Box>

                {
                    loading ? (
                        <Box sx={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                            gap: 2
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
                                        Drag and drop to start
                                    </Typography>
                                </Box>
                            ) : (
                                dateSections.map((section) => (
                                    <Box key={section.date} sx={{ mb: 6 }}>
                                        <Typography variant="h5" sx={{ mb: 3, fontWeight: 600, color: theme.palette.text.primary }}>
                                            {formatDate(section.date)}
                                        </Typography>

                                        {section.groups.map(group => (
                                            <Box key={group.id} sx={{ mb: 4, ml: 2 }}>
                                                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                                    <Typography variant="h6" sx={{ color: theme.palette.text.secondary, fontWeight: 500, mr: 2 }}>
                                                        {group.name}
                                                    </Typography>
                                                    <IconButton
                                                        size="small"
                                                        onClick={(e) => handleMenuOpen(e, group.id)}
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
                                                        // Convert DBImage to FileDetails for ImageCard
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
                                                                        if (imageUrls[img.id]) {
                                                                            setSelectedImage({ image: img, url: imageUrls[img.id] });
                                                                        }
                                                                    }}
                                                                />
                                                            </Box>
                                                        );
                                                    })}
                                                </Box>
                                            </Box>
                                        ))}
                                        <Divider sx={{ mt: 4 }} />
                                    </Box>
                                ))
                            )}
                        </Box>
                    )
                }
            </Box>

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
