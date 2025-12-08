import React from 'react';
import { Box, Skeleton } from '@mui/material';
import { useTheme } from '@mui/material/styles';

const MessageSkeleton: React.FC = () => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';

    return (
        <Box
            sx={{
                display: 'flex',
                justifyContent: 'flex-start',
                width: '100%',
                maxWidth: '70%',
            }}
        >
            <Box
                sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 1,
                    p: 2,
                    borderRadius: '16px',
                    background: isDark
                        ? 'rgba(255, 255, 255, 0.08)'
                        : 'rgba(0, 0, 0, 0.04)',
                    backdropFilter: 'blur(10px)',
                    WebkitBackdropFilter: 'blur(10px)',
                    border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
                }}
            >
                <Skeleton
                    variant="text"
                    width={200}
                    height={20}
                    sx={{
                        bgcolor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                    }}
                />
                <Skeleton
                    variant="text"
                    width={280}
                    height={20}
                    sx={{
                        bgcolor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                    }}
                />
                <Skeleton
                    variant="text"
                    width={150}
                    height={20}
                    sx={{
                        bgcolor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                    }}
                />
            </Box>
        </Box>
    );
};

export default MessageSkeleton;
