import { useState, useEffect } from 'react';
import { AppBar, Toolbar, Box, IconButton, Tooltip } from '@mui/material';
import { useTheme as useMuiTheme } from '@mui/material/styles';
import {
    Sidebar,
    Sun,
    Moon,
    Bell,
    House,
    SignOut,
    Gear,
    X,
    Minus,
    CornersOut,
    CornersIn
} from '@phosphor-icons/react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useMediaQuery } from '@mui/material';
import Breadcrumb from '../../../components/Breadcrumb';
import { useColorMode } from '../../../features/theme/ThemeContext';

interface NavbarProps {
    toggleLeftSidebar: () => void;
    toggleRightSidebar: () => void;
    leftSidebarOpen: boolean;
    rightSidebarOpen: boolean;
    agentIconShow: boolean;
}

export const NAVBAR_HEIGHT = 68;

const isElectron = () => {
    return !!(
        (window as any).process?.versions?.electron ||
        window.navigator.userAgent.toLowerCase().includes('electron') ||
        (window as any).windowControls
    );
};

export default function Navbar({
    toggleLeftSidebar,
    toggleRightSidebar,
    leftSidebarOpen,
    rightSidebarOpen,
}: NavbarProps) {
    const muiTheme = useMuiTheme();
    const { toggleColorMode } = useColorMode();
    const isDarkMode = muiTheme.palette.mode === 'dark';
    const location = useLocation();
    const navigate = useNavigate();
    const isMdUp = useMediaQuery(muiTheme.breakpoints.up('md'));
    const [isMaximized, setIsMaximized] = useState(false);

    const inElectron = isElectron();
    const windowControls = inElectron ? (window as any).windowControls : null;

    useEffect(() => {
        const checkMaximizeState = async () => {
            if (windowControls) {
                const maximized = await windowControls.isMaximized();
                setIsMaximized(maximized);
            }
        };

        checkMaximizeState();

        if (windowControls?.onStateChange) {
            windowControls.onStateChange((isMaximized: boolean) => {
                setIsMaximized(isMaximized);
            });
        }

        const handleResize = () => {
            checkMaximizeState();
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            if (windowControls?.removeStateChangeListener) {
                windowControls.removeStateChangeListener();
            }
        };
    }, [windowControls]);

    const handleMinimize = () => {
        if (windowControls) windowControls.minimize();
    };

    const handleMaximize = async () => {
        if (windowControls) {
            try {
                await windowControls.maximize();
                setTimeout(async () => {
                    const maximized = await windowControls.isMaximized();
                    setIsMaximized(maximized);
                }, 200);
            } catch (error) {
                console.error('Error in handleMaximize:', error);
            }
        }
    };

    const handleClose = () => {
        if (windowControls) windowControls.close();
    };

    let customBreadcrumbItems;
    if (location.pathname === '/') {
        customBreadcrumbItems = [{ label: 'Home', path: '/' }];
    } else {
        customBreadcrumbItems = undefined;
    }

    const leftSidebarWidth = leftSidebarOpen ? 212 : 0;
    const rightSidebarWidth = rightSidebarOpen ? 212 : 0;

    return (
        <AppBar
            position="fixed"
            color="default"
            sx={{
                boxShadow: 'none',
                borderBottom: `1px solid ${muiTheme.palette.divider}`,
                backgroundColor: muiTheme.palette.mode === 'light'
                    ? 'rgba(255, 255, 255, 0.3)'
                    : 'rgba(18, 18, 18, 0.3)',
                backdropFilter: 'blur(15px)',
                WebkitBackdropFilter: 'blur(15px)',
                ...(inElectron && {
                    WebkitAppRegion: 'drag',
                    userSelect: 'none',
                    cursor: 'default',
                }),
                left: { xs: 0, md: leftSidebarWidth },
                right: { xs: 0, md: rightSidebarWidth },
                width: {
                    xs: '100%',
                    md: `calc(100% - ${leftSidebarWidth}px - ${rightSidebarWidth}px)`
                },
                transition: (theme) => theme.transitions.create(['width', 'left', 'right'], {
                    easing: theme.transitions.easing.sharp,
                    duration: theme.transitions.duration.enteringScreen,
                }),
            }}
        >
            <Toolbar
                sx={{
                    display: 'flex',
                    width: '100%',
                    padding: { xs: '8px 8px', sm: '14px 28px' },
                    height: `${NAVBAR_HEIGHT}px`,
                    minHeight: `${NAVBAR_HEIGHT}px`,
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    flexWrap: { xs: 'wrap', sm: 'nowrap' },
                    gap: { xs: 1, sm: 0 },
                    ...(inElectron && { WebkitAppRegion: 'drag' }),
                }}
            >
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: { xs: '4px', sm: '8px' },
                        minWidth: 0,
                        flex: 1,
                        overflow: 'hidden',
                        ...(inElectron && { WebkitAppRegion: 'no-drag' }),
                    }}
                >
                    <IconButton
                        color="inherit"
                        aria-label="toggle sidebar"
                        onClick={toggleLeftSidebar}
                        sx={{ padding: 1, fontSize: { xs: 20, sm: 24 } }}
                    >
                        <Sidebar size={24} />
                    </IconButton>

                    <Tooltip title="Go to Home">
                        <IconButton
                            color="inherit"
                            aria-label="home"
                            onClick={() => navigate('/')}
                            sx={{ padding: 1, fontSize: { xs: 20, sm: 24 } }}
                        >
                            <House size={24} />
                        </IconButton>
                    </Tooltip>

                    {isMdUp && (
                        <Box sx={{ minWidth: 0, flex: 1, overflow: 'hidden' }}>
                            <Breadcrumb customItems={customBreadcrumbItems} />
                        </Box>
                    )}

                    {inElectron && (
                        <Box
                            sx={{
                                flexGrow: 1,
                                height: '100%',
                                minHeight: '40px',
                                WebkitAppRegion: 'drag',
                                cursor: 'default',
                                userSelect: 'none',
                                backgroundColor: 'transparent',
                                '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.02)' }
                            }}
                            title="Drag to move window"
                        />
                    )}
                </Box>

                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: { xs: '4px', sm: '8px' },
                        minWidth: 0,
                        flexShrink: 0,
                        ...(inElectron && { WebkitAppRegion: 'no-drag' }),
                    }}
                >
                    <Tooltip title={isDarkMode ? 'Light Mode' : 'Dark Mode'}>
                        <IconButton
                            color="inherit"
                            size="small"
                            onClick={toggleColorMode}
                            sx={{ fontSize: { xs: 18, sm: 20 } }}
                        >
                            {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
                        </IconButton>
                    </Tooltip>

                    <Tooltip title="Notifications">
                        <IconButton
                            color="inherit"
                            size="small"
                            onClick={toggleRightSidebar}
                            sx={{ fontSize: { xs: 18, sm: 20 } }}
                        >
                            <Bell size={20} />
                        </IconButton>
                    </Tooltip>

                    <Tooltip title="Settings">
                        <IconButton
                            color="inherit"
                            size="small"
                            onClick={() => navigate('/settings')}
                            sx={{ fontSize: { xs: 18, sm: 20 } }}
                        >
                            <Gear size={20} />
                        </IconButton>
                    </Tooltip>
                    <Tooltip title="Sign Out">
                        <IconButton
                            color="inherit"
                            size="small"
                            onClick={() => navigate('/auth')}
                        >
                            <SignOut size={20} />
                        </IconButton>
                    </Tooltip>

                    {inElectron && (
                        <Box sx={{ display: 'flex', gap: 0.5, ml: 1, WebkitAppRegion: 'no-drag' }}>
                            <Tooltip title="Minimize">
                                <IconButton
                                    color="inherit"
                                    size="small"
                                    onClick={handleMinimize}
                                    sx={{ fontSize: { xs: 16, sm: 18 }, WebkitAppRegion: 'no-drag', '&:hover': { backgroundColor: muiTheme.palette.action.hover } }}
                                >
                                    <Minus size={16} />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title={isMaximized ? 'Restore' : 'Maximize'}>
                                <IconButton
                                    color="inherit"
                                    size="small"
                                    onClick={handleMaximize}
                                    sx={{ fontSize: { xs: 16, sm: 18 }, WebkitAppRegion: 'no-drag', '&:hover': { backgroundColor: muiTheme.palette.action.hover } }}
                                >
                                    {isMaximized ? <CornersIn size={16} /> : <CornersOut size={16} />}
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Close">
                                <IconButton
                                    color="inherit"
                                    size="small"
                                    onClick={handleClose}
                                    sx={{ fontSize: { xs: 16, sm: 18 }, WebkitAppRegion: 'no-drag', '&:hover': { backgroundColor: muiTheme.palette.error.main, color: muiTheme.palette.error.contrastText } }}
                                >
                                    <X size={16} />
                                </IconButton>
                            </Tooltip>
                        </Box>
                    )}
                </Box>
            </Toolbar>
        </AppBar>
    );
}
