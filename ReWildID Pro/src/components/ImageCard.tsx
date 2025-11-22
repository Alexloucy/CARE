import React, { useEffect, useState, useRef } from 'react';
import { Box, Typography, Card, CardMedia, Fade, useTheme } from '@mui/material';
import { FileDetails } from '../types/electron';

interface ImageCardProps {
    file: FileDetails;
    date: string;
    loadImage: (date: string, path: string) => Promise<void>;
    imageUrl?: string;
    onClick: () => void;
}

const ImageCard: React.FC<ImageCardProps> = ({ file, date, loadImage, imageUrl, onClick }) => {
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

export default ImageCard;
