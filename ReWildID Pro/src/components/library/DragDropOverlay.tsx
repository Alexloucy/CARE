import React from 'react';
import { Box, Typography, Fade, useTheme } from '@mui/material';
import { UploadSimple } from '@phosphor-icons/react';

interface DragDropOverlayProps {
    isDragging: boolean;
}

export const DragDropOverlay: React.FC<DragDropOverlayProps> = ({ isDragging }) => {
    const theme = useTheme();

    return (
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
                <UploadSimple size={80} color={theme.palette.primary.main} weight="regular" />
                <Typography variant="h3" sx={{ mt: 4, fontWeight: 400, color: theme.palette.text.primary }}>
                    Drop to Upload
                </Typography>
            </Box>
        </Fade>
    );
};
