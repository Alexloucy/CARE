import React from 'react';
import { Box, Typography, Fade, Divider, Tooltip, IconButton, useTheme } from '@mui/material';
import { DownloadSimple, Trash } from '@phosphor-icons/react';

interface SelectionToolbarProps {
    selectedCount: number;
    onSave: () => void;
    onDelete: () => void;
    visible: boolean;
}

export const SelectionToolbar: React.FC<SelectionToolbarProps> = ({ selectedCount, onSave, onDelete, visible }) => {
    const theme = useTheme();

    return (
        <Fade in={visible}>
            <Box sx={{
                position: 'absolute',
                bottom: 32,
                left: '50%',
                transform: 'translateX(-50%)',
                bgcolor: theme.palette.mode === 'dark' ? 'rgba(30,30,30,0.9)' : 'rgba(255,255,255,0.9)',
                backdropFilter: 'blur(12px)',
                borderRadius: 4,
                boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
                p: 1.5,
                px: 3,
                display: 'flex',
                alignItems: 'center',
                gap: 3,
                zIndex: 100,
                border: `1px solid ${theme.palette.divider}`
            }}>
                <Typography variant="subtitle2" fontWeight="600">
                    {selectedCount} Selected
                </Typography>
                <Divider orientation="vertical" flexItem sx={{ height: 20, my: 'auto' }} />
                <Tooltip title="Save Selected">
                    <IconButton onClick={onSave} color="primary" size="small">
                        <DownloadSimple weight="bold" size={20} />
                    </IconButton>
                </Tooltip>
                <Tooltip title="Delete Selected">
                    <IconButton onClick={onDelete} color="error" size="small">
                        <Trash weight="bold" size={20} />
                    </IconButton>
                </Tooltip>
            </Box>
        </Fade>
    );
};
