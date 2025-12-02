import React, { useState, useRef, useEffect, useContext } from 'react';
import { Box, Typography, useTheme, ButtonBase, SxProps, Theme, TypographyProps } from '@mui/material';
import { SparkleIcon } from '@phosphor-icons/react';
import { AiModeContext } from '../contexts/AiModeContext';

interface AiModeButtonProps {
    text?: string;
    onClick?: (event: React.MouseEvent) => void;
    sx?: SxProps<Theme>;
    icon?: React.ReactNode;
    typographyProps?: TypographyProps;
    'data-tour'?: string;
}

const AiModeButton: React.FC<AiModeButtonProps> = ({ 
    text = "Detect", 
    onClick, 
    sx,
    icon = <SparkleIcon size={20} weight="fill" />,
    typographyProps,
    'data-tour': dataTour
}) => {
    const [renderPosition, setRenderPosition] = useState({ x: 0, y: 0 });
    const targetPosition = useRef({ x: 0, y: 0 });
    const [isHovered, setIsHovered] = useState(false);
    const [gradientRotation, setGradientRotation] = useState(0);
    const targetRotation = useRef(0);
    
    const aiContext = useContext(AiModeContext);
    // Fallback to local state if context is missing (though instructions implied global context is key)
    // But if context is missing, we default to false as per instructions "by default ... false"
    const shouldPlay = aiContext?.shouldPlayEffect ?? false;
    const setShouldPlay = aiContext?.setShouldPlayEffect ?? (() => {});

    const buttonRef = useRef<HTMLDivElement>(null);
    const theme = useTheme();

    // Intro Animation
    useEffect(() => {
        if (!shouldPlay) return;

        // Small delay to ensure layout is done
        const timer = setTimeout(() => {
            if (!buttonRef.current) return;
            const width = buttonRef.current.offsetWidth;
            const height = buttonRef.current.offsetHeight;
            
            let startTime: number | null = null;
            const duration = 1500; // ms

            const step = (timestamp: number) => {
                if (!startTime) startTime = timestamp;
                const progress = Math.min((timestamp - startTime) / duration, 1);
                
                // Animate from left to right
                setRenderPosition({
                    x: width * progress,
                    y: height / 2
                });

                if (progress < 1) {
                    // Check context again in case it was cancelled externally? 
                    // Actually we can just let it finish.
                    requestAnimationFrame(step);
                } else {
                    setShouldPlay(false);
                }
            };

            requestAnimationFrame(step);
        }, 500);

        return () => clearTimeout(timer);
    }, [shouldPlay, setShouldPlay]);

    // Stop intro on hover
    useEffect(() => {
        if (isHovered && shouldPlay) setShouldPlay(false);
    }, [isHovered, shouldPlay, setShouldPlay]);

    // Smooth mouse following effect
    useEffect(() => {
        if (shouldPlay) return; // Skip if intro is playing

        let animationFrameId: number;
        
        const animate = () => {
            setRenderPosition(prev => ({
                x: prev.x + (targetPosition.current.x - prev.x) * 0.04,
                y: prev.y + (targetPosition.current.y - prev.y) * 0.04
            }));
            // Smooth rotation animation
            setGradientRotation(prev => prev + (targetRotation.current - prev) * 0.04);
            animationFrameId = requestAnimationFrame(animate);
        };
        
        if (isHovered) {
            animate();
        }
        
        return () => cancelAnimationFrame(animationFrameId);
    }, [isHovered, shouldPlay]);

    const handleMouseMove = (e: React.MouseEvent) => {
        if (buttonRef.current) {
            const rect = buttonRef.current.getBoundingClientRect();
            const relativeY = e.clientY - rect.top;
            const height = rect.height;
            
            targetPosition.current = {
                x: e.clientX - rect.left,
                y: relativeY,
            };
            
            // Calculate rotation: 0deg at top, 180deg at bottom
            targetRotation.current = (relativeY / height) * 270;
            
            // If first entry, snap to position instantly to avoid "flying in" from 0,0
            if (!isHovered) {
                setRenderPosition(targetPosition.current);
                setGradientRotation(targetRotation.current);
            }
        }
    };

    // Colors matching the Google AI reference (Blue -> Red -> Yellow -> Green)
    // Rotation changes based on mouse Y position (0deg at top, 180deg at bottom)
    const gradient = `conic-gradient(from ${gradientRotation}deg at ${renderPosition.x}px ${renderPosition.y}px, 
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
            data-tour={dataTour}
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
                    opacity: isHovered || shouldPlay ? 1 : 0,
                    transition: 'opacity 0.4s ease', // Slightly slower fade for elegance
                    zIndex: 1,
                    maskImage: `radial-gradient(85px circle at ${renderPosition.x}px ${renderPosition.y}px, black, transparent)`,
                    WebkitMaskImage: `radial-gradient(85px circle at ${renderPosition.x}px ${renderPosition.y}px, black, transparent)`,
                }} 
            />

            {/* Inner Content with Ripple */}
            <ButtonBase 
                onClick={onClick}
                sx={{
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
                    ...sx as any
                }}
            >
                 {icon}
                 <Typography fontSize="0.9rem" {...typographyProps}>
                     {text}
                 </Typography>
            </ButtonBase>
        </Box>
    );
};

export default AiModeButton;