import React, { useEffect, useState, useRef, useMemo } from 'react';
import { Box, Typography, Card, CardMedia, Fade, useTheme, Skeleton } from '@mui/material';
import { FileDetails } from '../../types/electron';
import ImageModal from '../../components/ImageModal';
import { UploadSimple } from '@phosphor-icons/react';

interface DateGroup {
    date: string;
    files: FileDetails[];
}

// --- Image Card Component ---
const ImageCard: React.FC<{
    file: FileDetails;
    date: string;
    loadImage: (date: string, path: string) => Promise<void>;
    imageUrl?: string;
    onClick: () => void;
}> = ({ file, date, loadImage, imageUrl, onClick }) => {
    const theme = useTheme();
    const [showImage, setShowImage] = useState(false);
    const [isLoaded, setIsLoaded] = useState(false);
    const cardRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                setShowImage(entry.isIntersecting);
                if (!entry.isIntersecting) {
                    setIsLoaded(false);
                }
            },
            {
                rootMargin: '600px',
                threshold: 0
            }
        );

        if (cardRef.current) {
            observer.observe(cardRef.current);
        }

        return () => {
            observer.disconnect();
        };
    }, []);

    useEffect(() => {
        if (showImage && !imageUrl) {
            loadImage(date, file.path);
        }
    }, [showImage, imageUrl, date, file.path, loadImage]);

    return (
        <Card
            ref={cardRef}
            onClick={onClick}
            sx={{
                borderRadius: 3,
                overflow: 'hidden',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                aspectRatio: '1/1',
                position: 'relative',
                bgcolor: theme.palette.action.hover,
                cursor: 'pointer',
                group: 'true',
                '&:hover': {
                    boxShadow: '0 8px 24px rgba(0,0,0,0.2)',
                }
            }}
        >
            {imageUrl && showImage && (
                <Fade in={isLoaded} timeout={800}>
                    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
                        <CardMedia
                            component="img"
                            image={imageUrl}
                            alt={file.name}
                            onLoad={() => setIsLoaded(true)}
                            sx={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover',
                                transition: 'filter 0.3s ease',
                                '.MuiCard-root:hover &': {
                                    filter: 'brightness(0.85)'
                                }
                            }}
                        />

                        <Box sx={{
                            position: 'absolute',
                            inset: 0,
                            display: 'flex',
                            flexDirection: 'column',
                            justifyContent: 'space-between',
                            opacity: 0,
                            transition: 'opacity 0.3s ease',
                            '.MuiCard-root:hover &': {
                                opacity: 1
                            }
                        }}>
                            <Box sx={{
                                height: '40px',
                                background: 'linear-gradient(to bottom, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0) 100%)'
                            }} />

                            <Box sx={{
                                p: 1.5,
                                background: 'linear-gradient(to top, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0) 100%)',
                                color: 'white'
                            }}>
                                <Typography
                                    variant="body2"
                                    noWrap
                                    sx={{
                                        fontWeight: 500,
                                        textShadow: '0 1px 2px rgba(0,0,0,0.5)'
                                    }}
                                >
                                    {file.name}
                                </Typography>
                            </Box>
                        </Box>
                    </Box>
                </Fade>
            )}
        </Card>
    );
};

const LibraryPage: React.FC = () => {
    const [loading, setLoading] = useState(true);
    const [groups, setGroups] = useState<DateGroup[]>([]);
    const [imageUrls, setImageUrls] = useState<Record<string, string>>({});
    const [isDragging, setIsDragging] = useState(false);
    const theme = useTheme();

    // Modal State
    const [selectedFile, setSelectedFile] = useState<{ file: FileDetails, url: string, date: string } | null>(null);

    const fetchLibrary = async () => {
        try {
            setLoading(true);
            const response = await window.api.getImagePaths('');

            if (!response.ok || !response.selectAllPaths) {
                console.error('Failed to fetch library:', response.error);
                setLoading(false);
                return;
            }

            const paths = response.selectAllPaths;
            const groupsMap: Record<string, FileDetails[]> = {};

            paths.forEach(fullPath => {
                const normalizedPath = fullPath.replace(/\\/g, '/');
                const parts = normalizedPath.split('/');

                if (parts.length >= 2) {
                    const date = parts[0];
                    const relativePath = parts.slice(1).join('/');
                    const fileName = parts[parts.length - 1];

                    if (!groupsMap[date]) {
                        groupsMap[date] = [];
                    }

                    groupsMap[date].push({
                        name: fileName,
                        path: relativePath,
                        isDirectory: false
                    });
                }
            });

            const newGroups: DateGroup[] = Object.keys(groupsMap)
                .sort((a, b) => b.localeCompare(a))
                .map(date => ({
                    date: date,
                    files: groupsMap[date]
                }));

            setGroups(newGroups);
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

    const loadImage = async (date: string, path: string) => {
        const key = `${date}/${path}`;
        if (imageUrls[key]) return;

        try {
            const response = await window.api.viewImage(date, path);
            if (response.ok && response.data) {
                const blob = new Blob([response.data as unknown as BlobPart], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                setImageUrls(prev => ({ ...prev, [key]: url }));
            }
        } catch (error) {
            console.error(`Failed to load image ${path}:`, error);
        }
    };

    // Flatten files for navigation
    const allFiles = useMemo(() => {
        return groups.flatMap(group => group.files.map(file => ({ ...file, date: group.date })));
    }, [groups]);

    const handleNext = () => {
        if (!selectedFile) return;
        const currentIndex = allFiles.findIndex(f => f.path === selectedFile.file.path && f.date === selectedFile.date);
        if (currentIndex < allFiles.length - 1) {
            const nextFile = allFiles[currentIndex + 1];
            const key = `${nextFile.date}/${nextFile.path}`;

            if (!imageUrls[key]) {
                loadImage(nextFile.date, nextFile.path);
            }

            setSelectedFile({
                file: nextFile,
                url: imageUrls[key] || '',
                date: nextFile.date
            });
        }
    };

    const handlePrev = () => {
        if (!selectedFile) return;
        const currentIndex = allFiles.findIndex(f => f.path === selectedFile.file.path && f.date === selectedFile.date);
        if (currentIndex > 0) {
            const prevFile = allFiles[currentIndex - 1];
            const key = `${prevFile.date}/${prevFile.path}`;

            if (!imageUrls[key]) {
                loadImage(prevFile.date, prevFile.path);
            }

            setSelectedFile({
                file: prevFile,
                url: imageUrls[key] || '',
                date: prevFile.date
            });
        }
    };

    // Update selectedFile URL when it loads
    useEffect(() => {
        if (selectedFile && !selectedFile.url) {
            const key = `${selectedFile.date}/${selectedFile.file.path}`;
            if (imageUrls[key]) {
                setSelectedFile(prev => prev ? { ...prev, url: imageUrls[key] } : null);
            }
        }
    }, [imageUrls, selectedFile]);

    // Drag & Drop Handlers
    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = async (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        const files = Array.from(e.dataTransfer.files);
        if (files.length === 0) return;

        // Filter for images
        const imageFiles = files.filter(file => file.type.startsWith('image/') || file.name.toLowerCase().endsWith('.jpg') || file.name.toLowerCase().endsWith('.jpeg'));

        if (imageFiles.length === 0) {
            alert('Please upload valid image files (JPG).');
            return;
        }

        setLoading(true);
        let successCount = 0;

        for (const file of imageFiles) {
            try {
                const arrayBuffer = await file.arrayBuffer();
                const uint8Array = new Uint8Array(arrayBuffer);

                // Use file name as relative path for now (uploads to current date folder)
                const response = await window.api.uploadImage(file.name, uint8Array);

                if (response.ok) {
                    successCount++;
                } else {
                    console.error(`Failed to upload ${file.name}:`, response.error);
                }
            } catch (error) {
                console.error(`Error uploading ${file.name}:`, error);
            }
        }

        if (successCount > 0) {
            // Refresh library
            await fetchLibrary();
        }
        setLoading(false);
    };

    return (
        <Box
            sx={{
                p: 4,
                height: '100%',
                overflowY: 'auto',
                position: 'relative',
                outline: 'none'
            }}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
            {/* Drag Overlay */}
            <Fade in={isDragging}>
                <Box sx={{
                    position: 'absolute',
                    inset: 0,
                    zIndex: 100,
                    bgcolor: theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.8)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backdropFilter: 'blur(4px)',
                    border: `3px dashed ${theme.palette.primary.main}`,
                    borderRadius: 4,
                    m: 2
                }}>
                    <UploadSimple size={64} color={theme.palette.primary.main} />
                    <Typography variant="h5" sx={{ mt: 2, fontWeight: 600, color: theme.palette.primary.main }}>
                        Drop images to upload
                    </Typography>
                </Box>
            </Fade>

            <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h4" fontWeight="bold">Library</Typography>
            </Box>

            {loading ? (
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
                    {groups.length === 0 ? (
                        <Box
                            sx={{
                                mt: 10,
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                opacity: 0.6,
                                border: '2px dashed',
                                borderColor: 'divider',
                                borderRadius: 4,
                                p: 8
                            }}
                        >
                            <UploadSimple size={48} />
                            <Typography variant="h6" sx={{ mt: 2 }}>
                                No images yet
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                Drag and drop images here to get started
                            </Typography>
                        </Box>
                    ) : (
                        groups.map((group) => (
                            <Box key={group.date} sx={{ mb: 5 }}>
                                <Typography variant="h6" sx={{ mb: 2, color: 'text.secondary', fontWeight: 500 }}>
                                    {formatDate(group.date)}
                                </Typography>
                                <Box sx={{
                                    display: 'grid',
                                    gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                                    gap: 2
                                }}>
                                    {group.files.map((file) => {
                                        const key = `${group.date}/${file.path}`;
                                        return (
                                            <Box key={key}>
                                                <ImageCard
                                                    file={file}
                                                    date={group.date}
                                                    loadImage={loadImage}
                                                    imageUrl={imageUrls[key]}
                                                    onClick={() => {
                                                        if (imageUrls[key]) {
                                                            setSelectedFile({ file, url: imageUrls[key], date: group.date });
                                                        }
                                                    }}
                                                />
                                            </Box>
                                        );
                                    })}
                                </Box>
                            </Box>
                        ))
                    )}
                </Box>
            )}

            {/* Image Modal */}
            <ImageModal
                open={!!selectedFile}
                onClose={() => setSelectedFile(null)}
                imageUrl={selectedFile?.url}
                file={selectedFile?.file}
                onNext={handleNext}
                onPrev={handlePrev}
                hasNext={selectedFile ? allFiles.findIndex(f => f.path === selectedFile.file.path && f.date === selectedFile.date) < allFiles.length - 1 : false}
                hasPrev={selectedFile ? allFiles.findIndex(f => f.path === selectedFile.file.path && f.date === selectedFile.date) > 0 : false}
            />
        </Box>
    );
};

export default LibraryPage;
