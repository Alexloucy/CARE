import React, { useEffect, useState, useRef } from 'react';
import { Box, Typography, Grid, Card, CardMedia, CircularProgress, Fade, useTheme } from '@mui/material';
import { FileDetails } from '../../types/electron';

interface DateGroup {
    date: string;
    files: FileDetails[];
}

// Sub-component to handle individual image loading state
const ImageCard: React.FC<{
    file: FileDetails;
    date: string;
    loadImage: (date: string, path: string) => Promise<void>;
    imageUrl?: string;
}> = ({ file, date, loadImage, imageUrl }) => {
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
                rootMargin: '600px', // Load images well before they come into view
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
            sx={{
                borderRadius: 3,
                overflow: 'hidden',
                boxShadow: theme.palette.mode === 'dark' ? '0 4px 20px rgba(0,0,0,0.5)' : '0 2px 12px rgba(0,0,0,0.08)',
                transition: 'transform 0.2s',
                '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: theme.palette.mode === 'dark' ? '0 8px 30px rgba(0,0,0,0.6)' : '0 4px 20px rgba(0,0,0,0.12)',
                },
                aspectRatio: '1/1',
                position: 'relative',
                bgcolor: theme.palette.action.hover
            }}
        >
            {imageUrl && showImage && (
                <Fade in={isLoaded} timeout={800}>
                    <CardMedia
                        component="img"
                        image={imageUrl}
                        alt={file.name}
                        onLoad={() => setIsLoaded(true)}
                        sx={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                        }}
                    />
                </Fade>
            )}
        </Card>
    );
};

const LibraryPage: React.FC = () => {
    const [loading, setLoading] = useState(true);
    const [groups, setGroups] = useState<DateGroup[]>([]);
    const [imageUrls, setImageUrls] = useState<Record<string, string>>({});

    useEffect(() => {
        const fetchLibrary = async () => {
            try {
                setLoading(true);
                // 1. Fetch all image paths recursively
                const response = await window.api.getImagePaths('');

                if (!response.ok || !response.selectAllPaths) {
                    console.error('Failed to fetch library:', response.error);
                    setLoading(false);
                    return;
                }

                const paths = response.selectAllPaths;
                const groupsMap: Record<string, FileDetails[]> = {};

                // 2. Process paths to group by Date
                paths.forEach(fullPath => {
                    // Normalize path separators to forward slashes
                    const normalizedPath = fullPath.replace(/\\/g, '/');
                    const parts = normalizedPath.split('/');

                    // Expecting format: YYYYMMDD/subfolder/image.jpg or YYYYMMDD/image.jpg
                    if (parts.length >= 2) {
                        const date = parts[0];
                        // The image path relative to the date folder
                        // We need to reconstruct the path relative to the date folder for viewImage
                        // fullPath is relative to data/image_uploaded/1
                        // viewImage expects path relative to data/image_uploaded/1/{date}

                        // parts[0] is date. parts.slice(1) is the rest.
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

                // Convert map to array and sort by date descending
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

        fetchLibrary();
    }, []);

    // Helper to format date string (YYYYMMDD -> Readable)
    const formatDate = (dateStr: string) => {
        if (dateStr.length !== 8) return dateStr;
        const year = dateStr.substring(0, 4);
        const month = dateStr.substring(4, 6);
        const day = dateStr.substring(6, 8);
        const date = new Date(parseInt(year), parseInt(month) - 1, parseInt(day));
        return date.toLocaleDateString(undefined, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    };

    // Lazy load image content
    const loadImage = async (date: string, path: string) => {
        const key = `${date}/${path}`;
        if (imageUrls[key]) return;

        try {
            const response = await window.api.viewImage(date, path);
            if (response.ok && response.data) {
                // Cast to unknown then BlobPart to satisfy TS
                const blob = new Blob([response.data as unknown as BlobPart], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                setImageUrls(prev => ({ ...prev, [key]: url }));
            }
        } catch (error) {
            console.error(`Failed to load image ${path}:`, error);
        }
    };

    return (
        <Box sx={{ p: 4, height: '100%', overflowY: 'auto' }}>
            <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h4" fontWeight="bold">Library</Typography>
                {/* Placeholder for Upload Button */}
            </Box>

            {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 10 }}>
                    <CircularProgress />
                </Box>
            ) : (
                <Box>
                    {groups.length === 0 ? (
                        <Typography variant="body1" color="text.secondary" align="center" sx={{ mt: 10 }}>
                            No images found. Drag and drop folders to upload.
                        </Typography>
                    ) : (
                        groups.map((group) => (
                            <Box key={group.date} sx={{ mb: 5 }}>
                                <Typography variant="h6" sx={{ mb: 2, color: 'text.secondary', fontWeight: 500 }}>
                                    {formatDate(group.date)}
                                </Typography>
                                <Grid container spacing={2}>
                                    {group.files.map((file) => {
                                        const key = `${group.date}/${file.path}`;
                                        return (
                                            <Grid item xs={6} sm={4} md={3} lg={2} xl={2} key={key}>
                                                <ImageCard
                                                    file={file}
                                                    date={group.date}
                                                    loadImage={loadImage}
                                                    imageUrl={imageUrls[key]}
                                                />
                                            </Grid>
                                        );
                                    })}
                                </Grid>
                            </Box>
                        ))
                    )}
                </Box>
            )}
        </Box>
    );
};

export default LibraryPage;
