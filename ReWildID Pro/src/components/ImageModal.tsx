import React, { useEffect, useState, useRef } from 'react';
import { Box, Typography, Modal, IconButton, Fade, Backdrop, Paper, useTheme } from '@mui/material';
import { X, MagnifyingGlassPlus, MagnifyingGlassMinus, CaretLeft, CaretRight, Trash, Sparkle } from '@phosphor-icons/react';
import { FileDetails, Detection } from '../types/electron';
import { LiquidGlassOverlay } from './LiquidGlassOverlay';

// Toggle for ray-traced liquid glass (will be moved to settings later)
const USE_RAY_TRACED_GLASS = true;

// DetectionBox Component with fluid animations (1:1 copy of AiModeButton behavior)
interface DetectionBoxProps {
    bbox: { x: number; y: number; width: number; height: number };
    detection: Detection;
    zoom: number;
    containerWidth: number;
    containerHeight: number;
    useLiquidGlass?: boolean;
    onDelete?: (id: number) => void;
    customPopupContent?: React.ReactNode;
    popupTitle?: string;
    popupIcon?: React.ReactNode;
}

const DetectionBox: React.FC<DetectionBoxProps> = ({ 
    bbox, 
    detection, 
    containerWidth, 
    containerHeight, 
    useLiquidGlass = true,
    onDelete,
    customPopupContent,
    popupTitle = "Detection Details",
    popupIcon = <Sparkle size={18} weight="fill" color="#4285F4" />
}) => {
    const [isHovered, setIsHovered] = useState(false);
    const boxRef = useRef<HTMLDivElement>(null);
    const theme = useTheme();
    const hoverTimeout = useRef<NodeJS.Timeout | null>(null);

    // Local state for Liquid Glass animation (Start at center, then move to bbox)
    const [activeBbox, setActiveBbox] = useState(() => {
        if (useLiquidGlass) {
            // Initial Center Position
            const size = Math.min(containerWidth, containerHeight) * 0.2; // 20% of viewport or fixed
            return {
                x: (containerWidth / 2) - (size / 2),
                y: (containerHeight / 2) - (size / 2),
                width: size,
                height: size
            };
        }
        return bbox;
    });

    // Sync activeBbox with prop bbox for animation
    useEffect(() => {
        if (useLiquidGlass) {
            // Small delay to ensure the initial 'center' position is painted
            const timer = requestAnimationFrame(() => {
                 setActiveBbox(bbox);
            });
            return () => cancelAnimationFrame(timer);
        }
    }, [bbox, useLiquidGlass]);

    const isRightAligned = activeBbox.x + activeBbox.width + 240 > containerWidth;

    const handleMouseEnter = () => {
        if (hoverTimeout.current) clearTimeout(hoverTimeout.current);
        setIsHovered(true);
    };

    const handleMouseLeave = () => {
        hoverTimeout.current = setTimeout(() => {
            setIsHovered(false);
        }, 300);
    };

    // --- Standard BBox Style ---
    if (!useLiquidGlass) {
        return (
            <Box
                ref={boxRef}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
                sx={{
                    position: 'absolute',
                    left: bbox.x,
                    top: bbox.y,
                    width: bbox.width,
                    height: bbox.height,
                    border: '2px solid #4285F4',
                    backgroundColor: 'rgba(66, 133, 244, 0.1)',
                    cursor: 'pointer',
                    zIndex: 100,
                    transition: 'all 0.2s ease',
                    '&:hover': {
                        backgroundColor: 'rgba(66, 133, 244, 0.2)',
                        borderColor: '#5c9aff'
                    }
                }}
            >
                {/* Standard Label */}
                <Box
                    sx={{
                        position: 'absolute',
                        top: -24,
                        left: -2,
                        bgcolor: '#4285F4',
                        color: 'white',
                        fontSize: '12px',
                        fontWeight: 600,
                        px: 1,
                        py: 0.2,
                        borderRadius: '4px 4px 0 0',
                    }}
                >
                    {detection.label} ({Math.round(detection.confidence * 100)}%)
                </Box>
                
                {/* Info Popup for Standard Mode */}
                <Fade in={isHovered}>
                    <Paper sx={{
                        position: 'absolute',
                        top: '100%',
                        left: 0,
                        mt: 1,
                        p: 1.5,
                        zIndex: 101,
                        width: 200,
                        bgcolor: 'background.paper',
                        boxShadow: 3
                    }}>
                        <Typography variant="subtitle2" fontWeight="bold">{detection.label}</Typography>
                        <Typography variant="caption" display="block" color="text.secondary">
                            Confidence: {(detection.confidence * 100).toFixed(1)}%
                        </Typography>
                        {onDelete && (
                            <IconButton
                                size="small"
                                onClick={(e) => { e.stopPropagation(); onDelete(detection.id); }}
                                sx={{ mt: 1, color: '#ff4444', '&:hover': { bgcolor: 'rgba(255,68,68,0.1)' } }}
                            >
                                <Trash size={16} />
                                <Typography variant="caption" sx={{ ml: 0.5 }}>Delete Detection</Typography>
                            </IconButton>
                        )}
                    </Paper>
                </Fade>
            </Box>
        );
    }

    // --- Liquid Glass Style ---
    return (
        <Box
            ref={boxRef}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            sx={{
                position: 'absolute',
                left: activeBbox.x - (activeBbox.width * 0.03), // Center the expanded box
                top: activeBbox.y - (activeBbox.height * 0.03),
                width: activeBbox.width * 1.06, // Expand by 6%
                height: activeBbox.height * 1.06,
                borderRadius: '30px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                pointerEvents: 'auto',
                zIndex: 100,
                transition: 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)', // Slow, fluid movement for liquid glass
                bgcolor: 'rgba(255, 255, 255, 0.03)',
                
                // Hover State - Slight Light Up of Entire Glass
                '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.00)', // Subtle light up
                    '&::before': {
                        opacity: 0.15, // Boost rainbow visibility slightly
                    }
                },
                
                // Glass Rim (White highlights)
                boxShadow: `
                    inset 0 0 0 1px rgba(255, 255, 255, 0.15), 
                    inset 2px 2px 6px -2px rgba(255, 255, 255, 0.6), 
                    inset -2px -2px 6px -2px rgba(255, 255, 255, 0.2)
                `,
                
                // Rainbow Refraction Edge (::before)
                '&::before': {
                    content: '""',
                    position: 'absolute',
                    inset: 0,
                    zIndex: 1,
                    borderRadius: '30px',
                    padding: '1.5px', // Very thin thickness
                    
                    // Bright Rainbow Gradient
                    background: 'linear-gradient(125deg, #ff0000, #ff8800, #ffff00, #00ff00, #00ffff, #0000ff, #ff00ff)',
                    
                    // Masking magic to show only the border
                    mask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
                    maskComposite: 'exclude',
                    WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
                    WebkitMaskComposite: 'xor',
                    
                    opacity: 0.08, // Bright but slightly transparent
                    pointerEvents: 'none'
                },
                
                // Glass distortion layer (from .glassContainer::after) - ONLY AT EDGES
                '&::after': {
                    content: '""',
                    position: 'absolute',
                    zIndex: -1,
                    inset: 0,
                    borderRadius: '30px',
                    backdropFilter: 'blur(0px)',
                    filter: 'url(#container-glass)',
                    overflow: 'hidden',
                    isolation: 'isolate',
                    // Mask to create edge-only effect - transparent center, opaque edges
                    WebkitMask: 'radial-gradient(ellipse 80% 80% at center, transparent 60%, black 75%)',
                    mask: 'radial-gradient(ellipse 80% 80% at center, transparent 60%, black 75%)',
                }
            }}
        >
            {/* Content (Label Badge) - Elevated z-index to sit above glass */}
            <Box 
                sx={{ 
                    position: 'relative', 
                    width: '100%', 
                    height: '100%', 
                    zIndex: 3 
                }}
            >
                {/* Label Badge */}
                <Box
                    sx={{
                        position: 'absolute',
                        top: -35,
                        left: 7,
                        px: 1.5,
                        py: 0.5,
                        bgcolor: 'rgba(255, 255, 255, 0.25)',
                        color: 'white',
                        fontSize: '12px',
                        fontWeight: 600,
                        borderRadius: '12px',
                        whiteSpace: 'nowrap',
                        backdropFilter: 'blur(8px)',
                        border: '1px solid rgba(255, 255, 255, 0.3)',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                    }}
                >
                    {detection.label}
                </Box>
            </Box>

            {/* Info Popup */}
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
                        {customPopupContent ? (
                            customPopupContent
                        ) : (
                            <>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                                    {popupIcon}
                                    <Typography variant="subtitle2" fontWeight="700">
                                        {popupTitle}
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
                                    {onDelete && (
                                        <IconButton
                                            size="small"
                                            onClick={(e) => { e.stopPropagation(); onDelete(detection.id); }}
                                            sx={{ mt: 1, color: '#ff4444', width: '100%', borderRadius: 1, '&:hover': { bgcolor: 'rgba(255,68,68,0.1)' } }}
                                        >
                                            <Trash size={16} />
                                            <Typography variant="caption" sx={{ ml: 0.5 }}>Delete Detection</Typography>
                                        </IconButton>
                                    )}
                                </Box>
                            </>
                        )}
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
    useLiquidGlass?: boolean;
    onDeleteDetection?: (id: number) => void;
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
    detections,
    useLiquidGlass = true,
    onDeleteDetection
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
                    {/* SVG Filters for Liquid Glass Effect */}
                    <svg style={{ display: 'none' }}>
                        <filter id="container-glass" x="0%" y="0%" width="100%" height="100%">
                            <feTurbulence type="fractalNoise" baseFrequency="0.008 0.008" numOctaves="2" seed="92" result="noise" />
                            <feGaussianBlur in="noise" stdDeviation="0.02" result="blur" />
                            <feDisplacementMap in="SourceGraphic" in2="blur" scale="77" xChannelSelector="R" yChannelSelector="G" />
                        </filter>
                    </svg>

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
                            USE_RAY_TRACED_GLASS && imageUrl ? (
                                /* Ray-traced liquid glass - single WebGL canvas */
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
                                    <LiquidGlassOverlay
                                        imageUrl={imageUrl}
                                        bboxes={detections.map(det => {
                                            const bbox = transformBbox(det);
                                            return bbox ? { bbox, label: det.label } : null;
                                        }).filter(Boolean) as { bbox: { x: number; y: number; width: number; height: number }; label?: string }[]}
                                        containerWidth={imageDimensions.displayed.width}
                                        containerHeight={imageDimensions.displayed.height}
                                    />
                                </Box>
                            ) : (
                                /* CSS-based liquid glass - per detection */
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
                                                containerHeight={imageDimensions.displayed.height}
                                                useLiquidGlass={useLiquidGlass}
                                                onDelete={onDeleteDetection}
                                            />
                                        );
                                    })}
                                </Box>
                            )
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
export { DetectionBox };
export type { DetectionBoxProps };
