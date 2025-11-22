import React, { useEffect, useState, useRef } from 'react';
import { Box, Typography, Modal, IconButton, Fade, Backdrop, useTheme } from '@mui/material';
import { X, MagnifyingGlassPlus, MagnifyingGlassMinus, CaretLeft, CaretRight } from '@phosphor-icons/react';
import { FileDetails } from '../types/electron';

interface ImageModalProps {
    open: boolean;
    onClose: () => void;
    imageUrl?: string;
    file?: FileDetails;
    onNext?: () => void;
    onPrev?: () => void;
    hasNext?: boolean;
    hasPrev?: boolean;
}

const ImageModal: React.FC<ImageModalProps> = ({
    open,
    onClose,
    imageUrl,
    file,
    onNext,
    onPrev,
    hasNext,
    hasPrev
}) => {
    const [zoom, setZoom] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const dragStart = useRef({ x: 0, y: 0 });
    const theme = useTheme();

    // Reset state when opening a new image
    useEffect(() => {
        if (open) {
            setZoom(1);
            setPosition({ x: 0, y: 0 });
        }
    }, [open, imageUrl]); // Reset when imageUrl changes too

    // Keyboard Navigation
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (!open) return;
            if (e.key === 'ArrowRight' && hasNext && onNext) onNext();
            if (e.key === 'ArrowLeft' && hasPrev && onPrev) onPrev();
            if (e.key === 'Escape') onClose();
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [open, hasNext, hasPrev, onNext, onPrev, onClose]);

    const handleWheel = (e: React.WheelEvent) => {
        e.stopPropagation();
        if (e.deltaY < 0) {
            setZoom(prev => Math.min(prev + 0.1, 5));
        } else {
            setZoom(prev => Math.max(prev - 0.1, 0.5));
        }
    };

    // Drag Handlers
    const handleMouseDown = (e: React.MouseEvent) => {
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

    if (!file || !imageUrl) return null;

    return (
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
                    onClick={(e) => e.stopPropagation()}
                    sx={{
                        position: 'relative',
                        width: 'auto',
                        height: 'auto',
                        maxWidth: '90vw',
                        maxHeight: '90vh',
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
                        sx={{
                            position: 'relative',
                            flex: 1,
                            overflow: 'hidden',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            bgcolor: 'black',
                            minWidth: '400px',
                            minHeight: '300px',
                            cursor: isDragging ? 'grabbing' : 'grab'
                        }}
                        onWheel={handleWheel}
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseUp}
                    >
                        <img
                            src={imageUrl}
                            alt={file.name}
                            style={{
                                maxWidth: '100%',
                                maxHeight: '90vh',
                                objectFit: 'contain',
                                transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`,
                                transition: isDragging ? 'none' : 'transform 0.1s ease-out',
                                userSelect: 'none'
                            }}
                            draggable={false}
                            onDragStart={(e) => e.preventDefault()}
                        />

                        {/* Navigation Buttons (Overlay) */}
                        {hasPrev && (
                            <IconButton
                                onClick={(e) => { e.stopPropagation(); onPrev?.(); }}
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
                                onClick={(e) => { e.stopPropagation(); onNext?.(); }}
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
                            <IconButton
                                onClick={() => setZoom(z => Math.max(z - 0.5, 0.5))}
                                size="small"
                                sx={{ color: 'white', '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' } }}
                            >
                                <MagnifyingGlassMinus size={20} />
                            </IconButton>
                            <IconButton
                                onClick={() => setZoom(z => Math.min(z + 0.5, 5))}
                                size="small"
                                sx={{ color: 'white', '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' } }}
                            >
                                <MagnifyingGlassPlus size={20} />
                            </IconButton>
                            <IconButton
                                onClick={onClose}
                                size="small"
                                sx={{ color: 'white', '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' } }}
                            >
                                <X size={20} />
                            </IconButton>
                        </Box>

                        {/* Metadata Overlay (Bottom) */}
                        <Box sx={{
                            position: 'absolute',
                            bottom: 0,
                            left: 0,
                            right: 0,
                            p: 3,
                            background: 'linear-gradient(to top, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0) 100%)',
                            color: 'white',
                            pointerEvents: 'none'
                        }}>
                            <Typography variant="h6" sx={{ fontWeight: 600, textShadow: '0 2px 4px rgba(0,0,0,0.5)' }}>
                                {file.name}
                            </Typography>
                            <Typography variant="body2" sx={{ opacity: 0.8, textShadow: '0 1px 2px rgba(0,0,0,0.5)' }}>
                                {file.path}
                            </Typography>
                        </Box>
                    </Box>
                </Box>
            </Fade>
        </Modal>
    );
};

export default ImageModal;
