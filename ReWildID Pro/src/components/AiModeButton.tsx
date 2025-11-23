import React, { useState, useRef, useEffect } from 'react';
import { Box, Typography, useTheme, ButtonBase } from '@mui/material';
import { Sparkle } from '@phosphor-icons/react';

const AiModeButton: React.FC = () => {
    const [renderPosition, setRenderPosition] = useState({ x: 0, y: 0 });
    const targetPosition = useRef({ x: 0, y: 0 });
    const [isHovered, setIsHovered] = useState(false);
    const buttonRef = useRef<HTMLDivElement>(null);
    const theme = useTheme();

    // Smooth mouse following effect
    useEffect(() => {
        let animationFrameId: number;
        
        const animate = () => {
            setRenderPosition(prev => {
                // If very close, just snap (optimization could be added here, but constant lerp is smoother)
                // Speed factor 0.05 gives a more noticeable fluid delay (sluggish)
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
    }, [isHovered]);

    const handleMouseMove = (e: React.MouseEvent) => {
        if (buttonRef.current) {
            const rect = buttonRef.current.getBoundingClientRect();
            targetPosition.current = {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top,
            };
            // If first entry, snap to position instantly to avoid "flying in" from 0,0
            if (!isHovered) {
                setRenderPosition(targetPosition.current);
            }
        }
    };

    // Colors matching the Google AI reference (Blue -> Red -> Yellow -> Green)
    const gradient = `conic-gradient(from 0deg at ${renderPosition.x}px ${renderPosition.y}px, 
        #4285F4, 
        #9b72cb, 
        #d96570,
        #F4B400, 
        #0F9D58, 
        #4285F4
    )`;

    return (
        <Box
            ref={buttonRef}
            onMouseMove={handleMouseMove}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            sx={{
                position: 'relative',
                display: 'inline-flex',
                borderRadius: '9999px',
                cursor: 'pointer',
                overflow: 'hidden',
                p: '2px',
                // Removed scaling transform and box-shadow
            }}
        >
            {/* Base Border (Static) */}
            <Box 
                sx={{
                    position: 'absolute',
                    inset: 0,
                    borderRadius: '9999px',
                    bgcolor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0, 0, 0, 0.1)',
                    zIndex: 0
                }} 
            />

            {/* Gradient Spotlight Layer (Animated) */}
            <Box 
                sx={{
                    position: 'absolute',
                    inset: 0,
                    borderRadius: '9999px',
                    background: gradient,
                    opacity: isHovered ? 1 : 0,
                    transition: 'opacity 0.4s ease', // Slightly slower fade for elegance
                    zIndex: 1,
                    maskImage: `radial-gradient(65px circle at ${renderPosition.x}px ${renderPosition.y}px, black, transparent)`,
                    WebkitMaskImage: `radial-gradient(65px circle at ${renderPosition.x}px ${renderPosition.y}px, black, transparent)`,
                }} 
            />

            {/* Inner Content with Ripple */}
            <ButtonBase sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                px: 2.5,
                py: 1,
                borderRadius: '9999px',
                bgcolor: theme.palette.background.paper, 
                color: theme.palette.text.primary,
                position: 'relative',
                zIndex: 2, // Above borders
            }}>
                 <Sparkle size={20} weight="fill" />
                 <Typography fontWeight={600} fontSize="0.9rem">
                     AI Mode
                 </Typography>
            </ButtonBase>
        </Box>
    );
};

export default AiModeButton;
