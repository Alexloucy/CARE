import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Box, Modal, IconButton, Fade, Backdrop, Tooltip, Snackbar, CircularProgress } from '@mui/material';
import { X, MagnifyingGlassPlus, MagnifyingGlassMinus, Copy, CaretLeft, CaretRight } from '@phosphor-icons/react';

interface AgentImageModalProps {
    open: boolean;
    onClose: () => void;
    imageUrl: string | null;
    images?: string[];  // For gallery navigation
    initialIndex?: number;
}

const AgentImageModal: React.FC<AgentImageModalProps> = ({
    open,
    onClose,
    imageUrl,
    images,
    initialIndex = 0,
}) => {
    const [zoom, setZoom] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const dragStart = useRef({ x: 0, y: 0 });
    const [copySuccess, setCopySuccess] = useState(false);
    const [currentIndex, setCurrentIndex] = useState(initialIndex);
    const [imageLoaded, setImageLoaded] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    // Determine current image URL
    const currentImageUrl = images && images.length > 0
        ? images[currentIndex]
        : imageUrl;

    // Reset state when modal opens or image changes
    useEffect(() => {
        if (open) {
            setZoom(1);
            setPosition({ x: 0, y: 0 });
            setCurrentIndex(initialIndex);
            setImageLoaded(false);
        }
    }, [open, initialIndex]);

    // Reset zoom/pan when navigating to a new image in gallery
    useEffect(() => {
        setZoom(1);
        setPosition({ x: 0, y: 0 });
        setImageLoaded(false);
    }, [currentIndex]);

    // Handle keyboard navigation
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (!open) return;

            if (e.key === 'Escape') {
                onClose();
            } else if (e.key === 'ArrowRight' && images && currentIndex < images.length - 1) {
                setCurrentIndex(prev => prev + 1);
            } else if (e.key === 'ArrowLeft' && images && currentIndex > 0) {
                setCurrentIndex(prev => prev - 1);
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [open, onClose, images, currentIndex]);

    // Handle wheel zoom
    const handleWheel = (e: React.WheelEvent) => {
        e.stopPropagation();
        if (e.deltaY < 0) {
            setZoom(prev => Math.min(prev + 0.1, 5));
        } else {
            setZoom(prev => Math.max(prev - 0.1, 0.5));
        }
    };

    // Drag handlers for panning
    const handleMouseDown = (e: React.MouseEvent) => {
        // Only allow drag with left mouse button
        if (e.button !== 0) return;
        setIsDragging(true);
        dragStart.current = { x: e.clientX - position.x, y: e.clientY - position.y };
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (isDragging) {
            e.preventDefault();
            setPosition({
                x: e.clientX - dragStart.current.x,
                y: e.clientY - dragStart.current.y
            });
        }
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    // Copy image to clipboard using Electron's native API
    const handleCopy = useCallback(async () => {
        if (!currentImageUrl) return;

        try {
            // Use Electron's native clipboard API via IPC
            const result = await (window as any).api.copyImageToClipboard(currentImageUrl);
            if (result.success) {
                setCopySuccess(true);
            } else {
                console.error('Failed to copy image:', result.error);
            }
        } catch (error) {
            console.error('Failed to copy image:', error);
        }
    }, [currentImageUrl]);

    // Handle image load
    const handleImageLoad = () => {
        setImageLoaded(true);
    };

    if (!currentImageUrl) return null;

    const hasNext = images && currentIndex < images.length - 1;
    const hasPrev = images && currentIndex > 0;

    return (
        <>
            <Modal
                open={open}
                onClose={onClose}
                closeAfterTransition
                slots={{ backdrop: Backdrop }}
                slotProps={{
                    backdrop: {
                        timeout: 500,
                        sx: { backgroundColor: 'rgba(0, 0, 0, 0.85)' }
                    },
                }}
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    p: 4
                }}
            >
                <Fade in={open}>
                    <Box
                        onClick={(e: React.MouseEvent) => e.stopPropagation()}
                        sx={{
                            position: 'relative',
                            width: '90vw',
                            height: '90vh',
                            bgcolor: 'background.paper',
                            borderRadius: 4,
                            overflow: 'hidden',
                            boxShadow: 24,
                            display: 'flex',
                            flexDirection: 'column',
                            outline: 'none'
                        }}
                    >
                        {/* Image Container */}
                        <Box
                            ref={containerRef}
                            sx={{
                                position: 'relative',
                                flex: 1,
                                overflow: 'hidden',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                bgcolor: 'black',
                                cursor: isDragging ? 'grabbing' : 'grab'
                            }}
                            onWheel={handleWheel}
                            onMouseDown={handleMouseDown}
                            onMouseMove={handleMouseMove}
                            onMouseUp={handleMouseUp}
                            onMouseLeave={handleMouseUp}
                        >
                            {/* Loading indicator */}
                            {!imageLoaded && (
                                <Box
                                    sx={{
                                        position: 'absolute',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        width: '100%',
                                        height: '100%',
                                        zIndex: 5
                                    }}
                                >
                                    <CircularProgress size={40} sx={{ opacity: 0.7, color: 'white' }} />
                                </Box>
                            )}

                            {/* The image */}
                            <img
                                src={currentImageUrl}
                                alt="Preview"
                                onLoad={handleImageLoad}
                                style={{
                                    width: '100%',
                                    height: '100%',
                                    objectFit: 'contain',
                                    transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`,
                                    transition: isDragging ? 'none' : 'transform 0.1s ease-out',
                                    userSelect: 'none',
                                    opacity: imageLoaded ? 1 : 0
                                }}
                                draggable={false}
                                onDragStart={(e) => e.preventDefault()}
                            />

                            {/* Navigation arrows */}
                            {hasPrev && (
                                <IconButton
                                    onClick={(e: React.MouseEvent) => {
                                        e.stopPropagation();
                                        setCurrentIndex(prev => prev - 1);
                                    }}
                                    sx={{
                                        position: 'absolute',
                                        left: 16,
                                        top: '50%',
                                        transform: 'translateY(-50%)',
                                        color: 'white',
                                        bgcolor: 'rgba(0,0,0,0.4)',
                                        backdropFilter: 'blur(4px)',
                                        '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' },
                                        zIndex: 20
                                    }}
                                >
                                    <CaretLeft size={32} />
                                </IconButton>
                            )}
                            {hasNext && (
                                <IconButton
                                    onClick={(e: React.MouseEvent) => {
                                        e.stopPropagation();
                                        setCurrentIndex(prev => prev + 1);
                                    }}
                                    sx={{
                                        position: 'absolute',
                                        right: 16,
                                        top: '50%',
                                        transform: 'translateY(-50%)',
                                        color: 'white',
                                        bgcolor: 'rgba(0,0,0,0.4)',
                                        backdropFilter: 'blur(4px)',
                                        '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' },
                                        zIndex: 20
                                    }}
                                >
                                    <CaretRight size={32} />
                                </IconButton>
                            )}

                            {/* Top Toolbar (Floating) */}
                            <Box sx={{
                                position: 'absolute',
                                top: 16,
                                right: 16,
                                zIndex: 10,
                                display: 'flex',
                                gap: 1,
                                pointerEvents: 'auto',
                                bgcolor: 'rgba(0,0,0,0.4)',
                                borderRadius: 3,
                                p: 0.5,
                                backdropFilter: 'blur(4px)'
                            }}>
                                <Tooltip title="Copy Image">
                                    <IconButton
                                        onClick={handleCopy}
                                        size="small"
                                        sx={{ color: 'white', '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' } }}
                                    >
                                        <Copy size={20} />
                                    </IconButton>
                                </Tooltip>
                                <Tooltip title="Zoom Out">
                                    <IconButton
                                        onClick={() => setZoom(z => Math.max(z - 0.5, 0.5))}
                                        size="small"
                                        sx={{ color: 'white', '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' } }}
                                    >
                                        <MagnifyingGlassMinus size={20} />
                                    </IconButton>
                                </Tooltip>
                                <Tooltip title="Zoom In">
                                    <IconButton
                                        onClick={() => setZoom(z => Math.min(z + 0.5, 5))}
                                        size="small"
                                        sx={{ color: 'white', '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' } }}
                                    >
                                        <MagnifyingGlassPlus size={20} />
                                    </IconButton>
                                </Tooltip>
                                <Tooltip title="Close (Esc)">
                                    <IconButton
                                        onClick={onClose}
                                        size="small"
                                        sx={{ color: 'white', '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' } }}
                                    >
                                        <X size={20} />
                                    </IconButton>
                                </Tooltip>
                            </Box>

                            {/* Image counter for galleries */}
                            {images && images.length > 1 && (
                                <Box
                                    sx={{
                                        position: 'absolute',
                                        bottom: 16,
                                        left: '50%',
                                        transform: 'translateX(-50%)',
                                        bgcolor: 'rgba(0,0,0,0.6)',
                                        color: 'white',
                                        px: 2,
                                        py: 0.5,
                                        borderRadius: 2,
                                        fontSize: '0.875rem',
                                        backdropFilter: 'blur(4px)',
                                        zIndex: 10
                                    }}
                                >
                                    {currentIndex + 1} / {images.length}
                                </Box>
                            )}
                        </Box>
                    </Box>
                </Fade>
            </Modal>

            {/* Copy success notification */}
            <Snackbar
                open={copySuccess}
                autoHideDuration={2000}
                onClose={() => setCopySuccess(false)}
                message="Image copied to clipboard"
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
            />
        </>
    );
};

export default AgentImageModal;
