import React from 'react';
import { Box, Typography, useTheme } from '@mui/material';
import { Scan } from '@phosphor-icons/react';

const DetectionPage: React.FC = () => {
    const theme = useTheme();

    return (
        <Box sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
                <Typography variant="h4" fontWeight="bold">Detection</Typography>
            </Box>
            
            <Box 
                sx={{ 
                    height: '60vh', 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    color: theme.palette.text.secondary,
                    border: `2px dashed ${theme.palette.divider}`,
                    borderRadius: 3
                }}
            >
                <Scan size={64} weight="thin" />
                <Typography variant="h6" sx={{ mt: 2 }}>Detection Module Placeholder</Typography>
                <Typography variant="body2">Content coming soon...</Typography>
            </Box>
        </Box>
    );
};

export default DetectionPage;
