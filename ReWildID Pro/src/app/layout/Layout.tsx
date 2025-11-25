import { Box, useMediaQuery, useTheme } from '@mui/material';
import { useEffect, useState, useRef } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import LeftSidebar from './navbar/LeftSidebar';
import Navbar from './navbar/Navbar';
import { RightSidebar } from './navbar/RightSidebar';
import TaskPanel from '../../components/TaskPanel';

export default function Layout() {
    const location = useLocation();
    const theme = useTheme();
    const isDesktop = useMediaQuery(theme.breakpoints.up('md'));

    const [leftSidebarOpen, setLeftSidebarOpen] = useState(isDesktop);
    const [rightSidebarOpen, setRightSidebarOpen] = useState(false);

    const getPageMargin = (): number => {
        return 0
    };

    useEffect(() => {
        setLeftSidebarOpen(isDesktop);
        if (location.pathname.startsWith('/agent')) {
            setRightSidebarOpen(false);
        }
    }, [isDesktop, location.pathname]);

    // Auto-open right sidebar on new job
    const lastJobCount = useRef(0);
    useEffect(() => {
        const removeListener = window.api.onJobUpdate((jobs) => {
            if (jobs.length > lastJobCount.current) {
                 // New job added!
                 const latestJob = jobs[0]; 
                 if (latestJob.status === 'running' || latestJob.status === 'pending') {
                     setRightSidebarOpen(true);
                 }
            }
            lastJobCount.current = jobs.length;
        });
        return removeListener;
    }, []);

    return (
        <Box sx={{ display: 'flex', height: '100vh'}}>
            <LeftSidebar open={leftSidebarOpen} onClose={() => setLeftSidebarOpen(false)} />

            <Box sx={{
                display: 'flex',
                flexDirection: 'column',
                flexGrow: 1,
                overflow: 'hidden',
                width: {
                    xs: '100%',
                    md: `calc(100% - ${leftSidebarOpen ? 212 : 0}px - ${rightSidebarOpen ? 300 : 0}px)`
                },
                height: '100vh',
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

                <Box
                    component="main"
                    sx={{
                        mt: getPageMargin(),
                        mb: getPageMargin(),
                        flexGrow: 1,
                        overflow: 'auto',
                        paddingLeft: { xs: location.pathname === '/chat' ? 0 : 2, sm: 0 },
                        paddingRight: { xs: location.pathname === '/chat' ? 0 : 2, sm: 0 },
                        paddingTop: 0,
                        paddingBottom: 0,
                        // Modern thin scrollbar
                        '&::-webkit-scrollbar': {
                            width: '6px',
                        },
                        '&::-webkit-scrollbar-track': {
                            background: 'transparent',
                        },
                        '&::-webkit-scrollbar-thumb': {
                            background: theme.palette.mode === 'dark' 
                                ? 'rgba(255, 255, 255, 0.2)' 
                                : 'rgba(0, 0, 0, 0.2)',
                            borderRadius: '3px',
                            '&:hover': {
                                background: theme.palette.mode === 'dark' 
                                    ? 'rgba(255, 255, 255, 0.3)' 
                                    : 'rgba(0, 0, 0, 0.3)',
                            },
                        },
                        scrollbarWidth: 'thin',
                        scrollbarColor: theme.palette.mode === 'dark' 
                            ? 'rgba(255, 255, 255, 0.2) transparent' 
                            : 'rgba(0, 0, 0, 0.2) transparent',
                    }}
                >
                    <Outlet context={{ leftSidebarOpen, rightSidebarOpen }} />
                </Box>
            </Box>

            <RightSidebar
                open={rightSidebarOpen}
                onClose={() => setRightSidebarOpen(false)}
                title="Tasks"
            >
                <TaskPanel />
            </RightSidebar>
        </Box>
    );
}
