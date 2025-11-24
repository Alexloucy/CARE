import React, { useEffect, useState, useRef } from 'react';
import { Box, Typography, Modal, IconButton, Fade, Backdrop, Paper, useTheme } from '@mui/material';
import { X, MagnifyingGlassPlus, MagnifyingGlassMinus, CaretLeft, CaretRight, Trash, Sparkle } from '@phosphor-icons/react';
import { FileDetails, Detection } from '../types/electron';

// DetectionBox Component with fluid animations (1:1 copy of AiModeButton behavior)
interface DetectionBoxProps {
    bbox: { x: number; y: number; width: number; height: number };
    detection: Detection;
    zoom: number;
    containerWidth: number;
}

const DetectionBox: React.FC<DetectionBoxProps> = ({ bbox, detection, zoom, containerWidth }) => {
    const [renderPosition, setRenderPosition] = useState({ x: 0, y: 0 });
    const targetPosition = useRef({ x: 0, y: 0 });
    const [isHovered, setIsHovered] = useState(false);
    const [introActive, setIntroActive] = useState(true);
    const boxRef = useRef<HTMLDivElement>(null);
    const theme = useTheme();
    const hoverTimeout = useRef<NodeJS.Timeout | null>(null);

    const isRightAligned = bbox.x + bbox.width + 240 > containerWidth; // Popup width approx 240px

    // Intro Animation
    useEffect(() => {
        const timer = setTimeout(() => {
            if (!boxRef.current) return;
            const width = bbox.width;
            const height = bbox.height;
            
            let startTime: number | null = null;
            const duration = 1500;

            const step = (timestamp: number) => {
                if (!startTime) startTime = timestamp;
                const progress = Math.min((timestamp - startTime) / duration, 1);
                
                // Animate diagonal slide
                setRenderPosition({
                    x: width * progress,
                    y: height * progress
                });

                if (progress < 1) {
                    if (introActive) requestAnimationFrame(step);
                } else {
                    setIntroActive(false);
                }
            };

            requestAnimationFrame(step);
        }, 500);

        return () => clearTimeout(timer);
    }, [bbox.width, bbox.height]); // Re-run if size changes drastically

    // Stop intro on hover
    useEffect(() => {
        if (isHovered) setIntroActive(false);
    }, [isHovered]);

    // Smooth mouse following effect
    useEffect(() => {
        if (introActive) return;

        let animationFrameId: number;
        
        const animate = () => {
            setRenderPosition(prev => {
                return {
                    x: prev.x + (targetPosition.current.x - prev.x) * 0.05,
                    y: prev.y + (targetPosition.current.y - prev.y) * 0.05
                };
            });
            animationFrameId = requestAnimationFrame(animate);
        };
        
        if (isHovered) {
            animate();
        }
        
        return () => cancelAnimationFrame(animationFrameId);
    }, [isHovered, introActive]);

    const handleMouseMove = (e: React.MouseEvent) => {
        if (boxRef.current) {
            const rect = boxRef.current.getBoundingClientRect();
            // Calculate relative to the box (scaled by zoom)
            targetPosition.current = {
                x: (e.clientX - rect.left) / zoom, // Adjust for zoom scale on parent
                y: (e.clientY - rect.top) / zoom,
            };
            
            if (!isHovered) {
                setRenderPosition(targetPosition.current);
            }
        }
    };

    const handleMouseEnter = () => {
        if (hoverTimeout.current) clearTimeout(hoverTimeout.current);
        setIsHovered(true);
    };

    const handleMouseLeave = () => {
        hoverTimeout.current = setTimeout(() => {
            setIsHovered(false);
        }, 300); // 300ms grace period
    };

    const gradient = `conic-gradient(from 0deg at ${renderPosition.x}px ${renderPosition.y}px, 
        #2962FF, 
        #AA00FF, 
        #FF0055, 
        #FFD600, 
        #00C853, 
        #2962FF
    )`;

    return (
        <Box
            ref={boxRef}
            onMouseMove={handleMouseMove}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            sx={{
                position: 'absolute',
                left: bbox.x,
                top: bbox.y,
                width: bbox.width,
                height: bbox.height,
                pointerEvents: 'auto', // Enable interaction
            }}
        >
            {/* 0. Inner Light-up Fill (Hover Effect) */}
            <Box sx={{
                position: 'absolute',
                inset: 0,
                bgcolor: isHovered || introActive ? 'rgba(255, 255, 255, 0.07)' : 'transparent',
                transition: 'background-color 0.3s ease',
                borderRadius: 1.5,
                pointerEvents: 'none'
            }} />

            {/* 1. The Border Frame (Masked to be hollow) */}
            <Box sx={{
                position: 'absolute',
                inset: 0,
                p: '3px',
                borderRadius: 1.5,
                mask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
                maskComposite: 'exclude',
                WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
                WebkitMaskComposite: 'xor',
            }}>
                {/* A. Base Border (Glassy White/Grey - Brighter) */}
                <Box sx={{ 
                    position: 'absolute', 
                    inset: 0, 
                    bgcolor: 'rgba(255, 255, 255, 0.5)',
                }} />
                
                {/* B. Gradient Spotlight Border */}
                <Box sx={{
                    position: 'absolute',
                    inset: 0,
                    background: gradient,
                    opacity: isHovered || introActive ? 1 : 0,
                    transition: 'opacity 0.3s ease',
                    maskImage: `radial-gradient(${Math.min(bbox.width, bbox.height) * 1.5}px circle at ${renderPosition.x}px ${renderPosition.y}px, black, transparent)`,
                    WebkitMaskImage: `radial-gradient(${Math.min(bbox.width, bbox.height) * 1.5}px circle at ${renderPosition.x}px ${renderPosition.y}px, black, transparent)`,
                }} />
            </Box>

            {/* 2. Label Badge (Fixed above, Glassy) */}
            <Box
                sx={{
                    position: 'absolute',
                    top: -28,
                    left: 0,
                    px: 1.5,
                    py: 0.5,
                    bgcolor: 'rgba(255, 255, 255, 0.25)', // Semi-transparent white
                    color: 'white',
                    fontSize: '12px',
                    fontWeight: 600,
                    borderRadius: '12px', // Rounded
                    whiteSpace: 'nowrap',
                    backdropFilter: 'blur(8px)',
                    border: '1px solid rgba(255, 255, 255, 0.3)',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                }}
            >
                {detection.label}
            </Box>

            {/* 3. Info Popup (Anchored with smart positioning) */}
            <Fade in={isHovered}>
                <Box sx={{
                    position: 'absolute',
                    top: 0,
                    left: isRightAligned ? 'auto' : '100%',
                    right: isRightAligned ? '100%' : 'auto',
                    ml: isRightAligned ? 0 : 2,
                    mr: isRightAligned ? 2 : 0,
                    zIndex: 20,
                    // Invisible bridge to prevent closing when moving mouse
                    '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        [isRightAligned ? 'left' : 'right']: '100%',
                        width: '20px', // Bridge gap
                    }
                }}>
                    <Paper
                        sx={{
                            width: 240,
                            p: 2,
                            bgcolor: theme.palette.mode === 'dark' ? 'rgba(30, 30, 30, 0.7)' : 'rgba(255, 255, 255, 0.7)', // Glassy background
                            backdropFilter: 'blur(12px)',
                            borderRadius: 3,
                            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
                            border: `1px solid ${theme.palette.divider}`,
                            pointerEvents: 'auto',
                        }}
                    >
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                            <Sparkle size={18} weight="fill" color="#4285F4" />
                            <Typography variant="subtitle2" fontWeight="700">
                                Detection Details
                            </Typography>
                        </Box>
                        
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Typography variant="caption" color="text.secondary">Species</Typography>
                                <Box sx={{ 
                                    bgcolor: 'rgba(66, 133, 244, 0.1)', 
                                    color: '#4285F4', 
                                    px: 1, py: 0.2, 
                                    borderRadius: 1,
                                    fontSize: '0.75rem',
                                    fontWeight: 600
                                }}>
                                    {detection.label}
                                </Box>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption" color="text.secondary">Confidence</Typography>
                                <Typography variant="caption" fontWeight="600" sx={{ fontFamily: 'monospace' }}>
                                    {(detection.confidence * 100).toFixed(1)}%
                                </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                <Typography variant="caption" color="text.secondary">Detection Score</Typography>
                                <Typography variant="caption" fontWeight="600" sx={{ fontFamily: 'monospace' }}>
                                    {(detection.detection_confidence * 100).toFixed(1)}%
                                </Typography>
                            </Box>
                        </Box>
                    </Paper>
                </Box>
            </Fade>
        </Box>
    );
};

interface ImageModalProps {
    open: boolean;
    onClose: () => void;
    imageUrl?: string;
    file?: FileDetails;
    onNext?: () => void;
    onPrev?: () => void;
    hasNext?: boolean;
    hasPrev?: boolean;
    onDelete?: () => void;
    detections?: Detection[];
}

const ImageModal: React.FC<ImageModalProps> = ({
    open,
    onClose,
    imageUrl,
    file,
    onNext,
    onPrev,
    hasNext,
    hasPrev,
    onDelete,
    detections
}) => {
    const [zoom, setZoom] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const dragStart = useRef({ x: 0, y: 0 });
    const imageRef = useRef<HTMLImageElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [imageDimensions, setImageDimensions] = useState({ natural: { width: 0, height: 0 }, displayed: { width: 0, height: 0 } });

    // Reset state when opening a new image
    useEffect(() => {
        if (open) {
            setZoom(1);
            setPosition({ x: 0, y: 0 });
        }
    }, [open, imageUrl]); // Reset when imageUrl changes too

    // Calculate image dimensions when loaded
    const handleImageLoad = () => {
        if (imageRef.current && containerRef.current) {
            const img = imageRef.current;
            const container = containerRef.current;
            const containerRect = container.getBoundingClientRect();
            
            // Calculate displayed size (objectFit: contain)
            const imgAspect = img.naturalWidth / img.naturalHeight;
            const containerAspect = containerRect.width / containerRect.height;
            
            let displayedWidth, displayedHeight;
            if (imgAspect > containerAspect) {
                displayedWidth = containerRect.width;
                displayedHeight = containerRect.width / imgAspect;
            } else {
                displayedHeight = containerRect.height;
                displayedWidth = containerRect.height * imgAspect;
            }
            
            setImageDimensions({
                natural: { width: img.naturalWidth, height: img.naturalHeight },
                displayed: { width: displayedWidth, height: displayedHeight }
            });
        }
    };

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

    // Transform bbox coordinates from original image space to screen space
    const transformBbox = (detection: Detection) => {
        if (!containerRef.current || imageDimensions.natural.width === 0) return null;
        
        const { natural, displayed } = imageDimensions;
        const scale = displayed.width / natural.width;
        
        // Convert from absolute coordinates to scaled coordinates
        const x1 = detection.x1 * scale;
        const y1 = detection.y1 * scale;
        const x2 = detection.x2 * scale;
        const y2 = detection.y2 * scale;
        
        const width = x2 - x1;
        const height = y2 - y1;
        
        // Apply zoom and pan
        const containerRect = containerRef.current.getBoundingClientRect();
        const offsetX = (containerRect.width - displayed.width) / 2;
        const offsetY = (containerRect.height - displayed.height) / 2;
        
        return {
            x: offsetX + x1,
            y: offsetY + y1,
            width,
            height
        };
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
                        onMouseMove={(e) => {
                            handleMouseMove(e);
                        }}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={() => {
                            handleMouseUp();
                        }}
                    >
                        <img
                            ref={imageRef}
                            src={imageUrl}
                            alt={file.name}
                            onLoad={handleImageLoad}
                            style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'contain',
                                transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`,
                                transition: isDragging ? 'none' : 'transform 0.1s ease-out',
                                userSelect: 'none'
                            }}
                            draggable={false}
                            onDragStart={(e) => e.preventDefault()}
                        />

                        {/* Bounding Box Overlay */}
                        {detections && detections.length > 0 && imageDimensions.natural.width > 0 && (
                            <Box
                                sx={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    height: '100%',
                                    pointerEvents: 'none',
                                    transform: `scale(${zoom}) translate(${position.x / zoom}px, ${position.y / zoom}px)`,
                                    transition: isDragging ? 'none' : 'transform 0.1s ease-out'
                                }}
                            >
                                {detections.map((det, idx) => {
                                    const bbox = transformBbox(det);
                                    if (!bbox) return null;
                                    
                                    return (
                                        <DetectionBox 
                                            key={idx} 
                                            bbox={bbox} 
                                            detection={det} 
                                            zoom={zoom} 
                                            containerWidth={imageDimensions.displayed.width}
                                        />
                                    );
                                })}
                            </Box>
                        )}

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
                                onClick={(e) => {
                                    e.stopPropagation();
                                    if (window.confirm('Are you sure you want to delete this image?')) {
                                        onDelete?.();
                                    }
                                }}
                                size="small"
                                sx={{ color: '#ff4444', '&:hover': { bgcolor: 'rgba(255,68,68,0.2)' }, mr: 1 }}
                            >
                                <Trash size={20} />
                            </IconButton>
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
