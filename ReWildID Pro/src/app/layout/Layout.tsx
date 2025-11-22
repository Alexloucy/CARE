import { Box, useMediaQuery, useTheme } from '@mui/material';
import { useEffect, useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import LeftSidebar from './navbar/LeftSidebar';
import Navbar, { NAVBAR_HEIGHT } from './navbar/Navbar';
import { RightSidebar } from './navbar/RightSidebar';

export default function Layout() {
    const location = useLocation();
    const theme = useTheme();
    const isDesktop = useMediaQuery(theme.breakpoints.up('md'));

    const [leftSidebarOpen, setLeftSidebarOpen] = useState(isDesktop);
    const [rightSidebarOpen, setRightSidebarOpen] = useState(false);

    const getPageMargin = (pathname: string): number => {
        const exactPaths = ['/', '/chat'];
        if (exactPaths.includes(pathname)) return 0;
        if (pathname.startsWith('/admin')) return 0;
        if (pathname.startsWith('/agent')) return 0;
        return 4;
    };

    useEffect(() => {
        setLeftSidebarOpen(isDesktop);
        if (location.pathname.startsWith('/agent')) {
            setRightSidebarOpen(false);
        }
    }, [isDesktop, location.pathname]);

    return (
        <Box sx={{ display: 'flex' }}>
            <LeftSidebar open={leftSidebarOpen} onClose={() => setLeftSidebarOpen(false)} />

            <Box sx={{
                display: 'flex',
                flexDirection: 'column',
                flexGrow: 1,
                width: {
                    xs: '100%',
                    md: `calc(100% - ${leftSidebarOpen ? 212 : 0}px - ${rightSidebarOpen ? 212 : 0}px)`
                },
                ml: { xs: 0, md: leftSidebarOpen ? 0 : 0 },
                transition: theme => theme.transitions.create(['width', 'margin'], {
                    easing: theme.transitions.easing.sharp,
                    duration: theme.transitions.duration.enteringScreen,
                })
            }}>
                <Navbar
                    toggleLeftSidebar={() => setLeftSidebarOpen(!leftSidebarOpen)}
                    toggleRightSidebar={() => setRightSidebarOpen(!rightSidebarOpen)}
                    leftSidebarOpen={leftSidebarOpen}
                    rightSidebarOpen={rightSidebarOpen}
                    agentIconShow={!location.pathname.startsWith('/agent')}
                />

                <Box sx={{ height: `${NAVBAR_HEIGHT}px` }} />

                <Box
                    component="main"
                    sx={{
                        mt: getPageMargin(location.pathname),
                        mb: getPageMargin(location.pathname),
                        flexGrow: 1,
                        paddingLeft: { xs: location.pathname === '/chat' ? 0 : 2, sm: 0 },
                        paddingRight: { xs: location.pathname === '/chat' ? 0 : 2, sm: 0 },
                        paddingTop: 0,
                        paddingBottom: 0,
                    }}
                >
                    <Outlet />
                </Box>
            </Box>

            <RightSidebar
                open={rightSidebarOpen}
                onClose={() => setRightSidebarOpen(false)}
                title="Notifications"
            >
                <Box p={2}>
                    Notifications content here.
                </Box>
            </RightSidebar>
        </Box>
    );
}
