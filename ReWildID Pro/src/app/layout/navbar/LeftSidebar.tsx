import {
    Avatar,
    Box,
    Drawer,
    List,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    styled,
    Typography,
    useMediaQuery,
    useTheme,
} from '@mui/material';
import {
    House,
    ArrowSquareIn,
    Images,
    MagnifyingGlass,
    Gear,
} from '@phosphor-icons/react';
import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { alpha } from '@mui/material/styles';

interface LeftSidebarProps {
    open: boolean;
    onClose: () => void;
}

const DRAWER_WIDTH = 212;

const NavText = styled(Typography)({
    fontFamily: 'Inter, sans-serif',
    fontSize: '14px',
    fontStyle: 'normal',
    fontWeight: 400,
    lineHeight: '20px',
    letterSpacing: '0px',
    fontFeatureSettings: "'ss01' on, 'cv01' on"
});

const isElectron = () => {
    return !!(
        (window as any).process?.versions?.electron ||
        window.navigator.userAgent.toLowerCase().includes('electron') ||
        (window as any).windowControls
    );
};

// Mock user for display
const MOCK_USER = {
    displayName: 'Demo User',
    username: 'demo',
    profilePicture: null
};

export const LeftSidebar: React.FC<LeftSidebarProps> = ({ open, onClose }) => {
    const theme = useTheme();
    const location = useLocation();
    const navigate = useNavigate();
    const isMobile = useMediaQuery(theme.breakpoints.down('md'));

    const navItems = [
        { key: 'dashboard', label: 'Dashboard', path: '/', icon: <House weight="regular" size={20} /> },
        { key: 'import', label: 'Import', path: '/import', icon: <ArrowSquareIn weight="regular" size={20} /> },
        { key: 'library', label: 'Library', path: '/library', icon: <Images weight="regular" size={20} /> },
        { key: 'analysis', label: 'Analysis', path: '/analysis', icon: <MagnifyingGlass weight="regular" size={20} /> },
        { key: 'settings', label: 'Settings', path: '/settings', icon: <Gear weight="regular" size={20} /> },
    ];

    const isActivePath = (path: string) => {
        if (path === '/') return location.pathname === path;
        return location.pathname.startsWith(path);
    };

    const handleNavigate = (path: string) => {
        navigate(path);
        if (isMobile) onClose();
    };

    return (
        <Drawer
            variant={isMobile ? 'temporary' : 'persistent'}
            open={open}
            onClose={onClose}
            sx={{
                ...(isElectron() && { WebkitUserSelect: 'none', WebkitAppRegion: 'drag' }),
                width: open ? DRAWER_WIDTH : 0,
                flexShrink: 0,
                overflowX: 'hidden',
                '& .MuiDrawer-paper': {
                    width: DRAWER_WIDTH,
                    boxSizing: 'border-box',
                    borderRight: isMobile ? 'none' : '1px solid',
                    borderColor: 'divider',
                    bgcolor: theme.palette.mode === 'light' ? '#F9F9F9' : theme.palette.background.paper,
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    padding: '16px',
                    gap: '8px',
                    transition: theme.transitions.create('width', {
                        easing: theme.transitions.easing.sharp,
                        duration: theme.transitions.duration.enteringScreen,
                    }),
                },
                transition: theme.transitions.create('width', {
                    easing: theme.transitions.easing.sharp,
                    duration: theme.transitions.duration.leavingScreen,
                }),
            }}
            ModalProps={{ keepMounted: true }}
        >
            <Box
                sx={{
                    ...(isElectron() && { WebkitAppRegion: 'no-drag' }),
                    display: 'flex',
                    alignItems: 'center',
                    width: '100%',
                    padding: '8px',
                    borderRadius: '12px',
                    mb: 2,
                    gap: '8px',
                    border: 'none',
                    background: 'none',
                    cursor: 'default',
                    textAlign: 'left',
                }}
            >
                <Avatar sx={{ width: 32, height: 32, bgcolor: theme.palette.primary.main }}>
                    {MOCK_USER.displayName.charAt(0)}
                </Avatar>
                <NavText noWrap color={theme.palette.text.primary} sx={{ ml: 1 }}>
                    {MOCK_USER.displayName}
                </NavText>
            </Box>

            <List sx={{ width: '100%', p: 0 }}>
                {navItems.map((item) => {
                    const isActive = isActivePath(item.path);

                    return (
                        <Box key={item.key} sx={{ position: 'relative' }}>
                            <ListItemButton
                                selected={isActive}
                                onClick={() => handleNavigate(item.path)}
                                sx={{
                                    ...(isElectron() && { WebkitAppRegion: 'no-drag' }),
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    padding: '8px',
                                    borderRadius: '12px',
                                    mb: 0.5,
                                    '&.Mui-selected': {
                                        bgcolor: theme.palette.mode === 'dark' ? alpha(theme.palette.common.white, 0.08) : alpha(theme.palette.common.black, 0.05),
                                        color: theme.palette.text.primary,
                                        fontWeight: 500,
                                        '&:hover': {
                                            bgcolor: theme.palette.mode === 'dark' ? alpha(theme.palette.common.white, 0.12) : alpha(theme.palette.common.black, 0.08),
                                        }
                                    },
                                    '&:hover': {
                                        bgcolor: theme.palette.mode === 'dark' ? alpha(theme.palette.common.white, 0.05) : alpha(theme.palette.common.black, 0.04),
                                        '&:hover .sidebar-item-menu-button': { opacity: 1 },
                                    },
                                }}
                            >
                                <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
                                    <ListItemIcon sx={{ minWidth: 36, color: isActive ? theme.palette.text.primary : theme.palette.text.secondary, overflow: 'visible' }}>
                                        {item.icon}
                                    </ListItemIcon>
                                    <ListItemText
                                        primary={item.label}
                                        primaryTypographyProps={{
                                            fontFamily: 'Inter, sans-serif',
                                            fontSize: '14px',
                                            fontWeight: isActive ? 500 : 400,
                                            lineHeight: '20px',
                                            letterSpacing: '0px',
                                            sx: { fontFeatureSettings: "'ss01' on, 'cv01' on" },
                                            color: isActive ? theme.palette.text.primary : theme.palette.text.secondary
                                        }}
                                    />
                                </Box>
                            </ListItemButton>
                        </Box>
                    );
                })}
            </List>

            <Box sx={{ mt: 'auto', display: 'flex', flexDirection: 'column', alignItems: 'center', p: 2 }}>
                <Typography variant="h6" sx={{ fontFamily: 'Inter, sans-serif', fontWeight: 500, color: theme.palette.mode === 'dark' ? 'white' : '#1C1C1C', opacity: 0.7, fontSize: '16px' }}>
                    RewildID Pro
                </Typography>
            </Box>

        </Drawer>
    );
};

export default LeftSidebar;
