import React from 'react';
import {
    Box,
    Drawer,
    Typography,
    useTheme,
    useMediaQuery,
    IconButton
} from '@mui/material';
import { X } from '@phosphor-icons/react';

interface RightSidebarProps {
    open: boolean;
    onClose: () => void;
    title?: string;
    children?: React.ReactNode;
}

const DRAWER_WIDTH = 212;

export const RightSidebar: React.FC<RightSidebarProps> = ({
    open,
    onClose,
    title = '',
    children
}) => {
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('md'));

    return (
        <Drawer
            variant={isMobile ? 'temporary' : 'persistent'}
            anchor="right"
            open={open}
            onClose={onClose}
            sx={{
                width: isMobile ? (open ? DRAWER_WIDTH : 0) : (open ? DRAWER_WIDTH : 0),
                flexShrink: 0,
                '& .MuiDrawer-paper': {
                    width: DRAWER_WIDTH,
                    boxSizing: 'border-box',
                    borderLeft: '1px solid',
                    borderColor: 'divider',
                    backgroundColor: theme.palette.mode === 'dark' ? '#121212' : theme.palette.background.default,
                    transition: theme.transitions.create('width', {
                        easing: theme.transitions.easing.sharp,
                        duration: theme.transitions.duration.shorter,
                    }),
                },
            }}
            ModalProps={{
                keepMounted: true,
            }}
        >
            <Box
                sx={{
                    height: '68px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    px: 2,
                    backgroundColor: theme.palette.mode === 'dark' ? '#121212' : theme.palette.background.default,
                }}
            >
                <Typography variant="h6" sx={{ fontWeight: 500 }}>
                    {title || 'Notifications'}
                </Typography>

                {isMobile && (
                    <IconButton
                        onClick={onClose}
                        size="small"
                        aria-label="Close"
                    >
                        <X size={20} />
                    </IconButton>
                )}
            </Box>
            <Box sx={{
                p: 2,
                backgroundColor: theme.palette.mode === 'dark' ? '#121212' : theme.palette.background.default,
                height: '100%'
            }}>
                {children || (
                    <Typography color="text.secondary">
                        No notifications.
                    </Typography>
                )}
            </Box>
        </Drawer>
    );
};

export default RightSidebar;
