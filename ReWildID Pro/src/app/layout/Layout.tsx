import { Box, useMediaQuery, useTheme } from '@mui/material';
import { useEffect, useState, useRef, useCallback } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import LeftSidebar from './navbar/LeftSidebar';
import Navbar from './navbar/Navbar';
import { RightSidebar } from './navbar/RightSidebar';
import TaskPanel from '../../components/TaskPanel';
import { DragDropOverlay } from '../../components/library/DragDropOverlay';
import { UploadActionDialog } from '../../components/UploadActionDialog';
import { useColorMode } from '../../features/theme/ThemeContext';
import PrismaticBurst from '../../components/backgrounds/PrismaticBurst';
import ColorBends from '../../components/backgrounds/ColorBends';
import FloatingLines from '../../components/backgrounds/FloatingLines';
import Galaxy from '../../components/backgrounds/Galaxy';
import LightPillar from '../../components/backgrounds/LightPillar';

export default function Layout() {
    const location = useLocation();
    const navigate = useNavigate();
    const theme = useTheme();
    const { colorTheme } = useColorMode();
    const isDesktop = useMediaQuery(theme.breakpoints.up('md'));

    const [leftSidebarOpen, setLeftSidebarOpen] = useState(isDesktop);
    const [rightSidebarOpen, setRightSidebarOpen] = useState(false);

    // Global Drag & Drop State
    const [isDragging, setIsDragging] = useState(false);
    const [pendingFiles, setPendingFiles] = useState<string[]>([]);
    const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
    const dragCounter = useRef(0);

    const getPageMargin = (): number => {
        return 0
    };

    // Global Drag & Drop Handlers
    const handleDragEnter = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        dragCounter.current++;
        if (e.dataTransfer.types.includes('Files')) {
            setIsDragging(true);
        }
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        dragCounter.current--;
        if (dragCounter.current === 0) {
            setIsDragging(false);
        }
    }, []);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        dragCounter.current = 0;
        setIsDragging(false);

        const files = Array.from(e.dataTransfer.files);
        if (files.length === 0) return;

        const paths = files.map(file => window.api.getPathForFile(file));
        setPendingFiles(paths);
        setUploadDialogOpen(true);
    }, []);

    // Global upload button handler - opens file dialog then shows upload dialog
    const handleGlobalUploadClick = useCallback(async () => {
        const result = await window.api.openFileDialog();
        if (!result.canceled && result.filePaths.length > 0) {
            setPendingFiles(result.filePaths);
            setUploadDialogOpen(true);
        }
    }, []);

    // Listen for global upload trigger events
    useEffect(() => {
        const handleUploadEvent = () => handleGlobalUploadClick();
        window.addEventListener('trigger-upload', handleUploadEvent);
        return () => window.removeEventListener('trigger-upload', handleUploadEvent);
    }, [handleGlobalUploadClick]);

    // Handle upload confirmation from dialog
    const handleUploadConfirm = useCallback(async (
        action: 'library' | 'classify' | 'reid',
        groupName?: string,
        species?: string
    ) => {
        try {
            // Pass action and species to backend for chained processing
            const afterAction = action === 'library' ? undefined : action;
            await window.api.uploadPaths(pendingFiles, groupName, afterAction, species);
            setPendingFiles([]);

            // Navigate to appropriate page based on action
            if (action === 'classify') {
                navigate('/classification');
            } else if (action === 'reid') {
                navigate('/reid');
            } else {
                navigate('/library');
            }
        } catch (error) {
            console.error('Upload failed:', error);
        }
    }, [pendingFiles, navigate]);

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

    const hasGradient = colorTheme.gradient !== 'none' || !!colorTheme.special || !!colorTheme.image;

    return (
        <Box sx={{ position: 'relative', height: '100vh', overflow: 'hidden' }}>
            {/* Background Layer - Gradient or Special Animated */}
            {colorTheme.special === 'prismatic-burst' ? (
                <Box
                    sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        zIndex: 0,
                        bgcolor: '#0a0a0a',
                    }}
                >
                    <PrismaticBurst
                        animationType="rotate3d"
                        intensity={2}
                        speed={0.15}
                        distort={1.0}
                        paused={false}
                        offset={{ x: 0, y: 0 }}
                        hoverDampness={0.25}
                        rayCount={24}
                        mixBlendMode="lighten"
                        colors={['#ff007a', '#4d3dff', '#ffffff']}
                    />
                </Box>
            ) : colorTheme.special === 'color-bends' ? (
                <Box
                    sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        zIndex: 0,
                        bgcolor: '#0a0a0a',
                    }}
                >
                    <ColorBends
                        colors={['#ff5c7a', '#8a5cff', '#00ffd1']}
                        rotation={30}
                        speed={0.3}
                        scale={1}
                        frequency={1.4}
                        warpStrength={1.2}
                        mouseInfluence={0.8}
                        parallax={0.6}
                        noise={0.08}
                        transparent={false}
                    />
                </Box>
            ) : colorTheme.special === 'floating-lines' ? (
                <Box
                    sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        zIndex: 0,
                        bgcolor: '#0a0a0a',
                    }}
                >
                    <FloatingLines
                        enabledWaves={['top', 'middle', 'bottom']}
                        lineCount={[10, 15, 20]}
                        lineDistance={[8, 6, 4]}
                        bendRadius={5.0}
                        bendStrength={-0.5}
                        interactive={true}
                        parallax={true}
                    />
                </Box>
            ) : colorTheme.special === 'galaxy' ? (
                <Box
                    sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        zIndex: 0,
                        bgcolor: '#0a0a0a',
                    }}
                >
                    <Galaxy
                        mouseRepulsion={true}
                        mouseInteraction={true}
                        density={1.5}
                        glowIntensity={0.5}
                        saturation={0.8}
                        hueShift={240}
                        speed={0.3}
                        rotationSpeed={0.02}
                        transparent={false}
                    />
                </Box>
            ) : colorTheme.special === 'light-pillar' ? (
                <Box
                    sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        zIndex: 0,
                        bgcolor: '#0a0a0a',
                    }}
                >
                    <LightPillar
                        topColor="#5227FF"
                        bottomColor="#FF9FFC"
                        intensity={1.0}
                        rotationSpeed={0.15}
                        glowAmount={0.005}
                        pillarWidth={3.0}
                        pillarHeight={0.4}
                        noiseIntensity={0.5}
                        pillarRotation={0}
                        interactive={false}
                        mixBlendMode="normal"
                    />
                </Box>
            ) : colorTheme.image ? (
                <Box
                    sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        zIndex: 0,
                        backgroundImage: `url(${colorTheme.image})`,
                        backgroundSize: 'cover',
                        backgroundPosition: 'center',
                        backgroundRepeat: 'no-repeat',
                    }}
                />
            ) : hasGradient && (
                <Box
                    sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        background: colorTheme.gradient,
                        zIndex: 0,
                        transition: 'background 0.3s ease-in-out',
                    }}
                />
            )}

            {/* Main Content Layer */}
            <Box
                sx={{
                    display: 'flex',
                    height: '100vh',
                    position: 'relative',
                    zIndex: 1,
                }}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
            >
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
                    transition: (theme: import('@mui/material').Theme) => theme.transitions.create(['width', 'margin'], {
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

                {/* Global Drag & Drop Overlay */}
                <DragDropOverlay isDragging={isDragging} />

                {/* Upload Action Dialog - handles both action selection and group naming */}
                <UploadActionDialog
                    open={uploadDialogOpen}
                    onClose={() => {
                        setUploadDialogOpen(false);
                        setPendingFiles([]);
                    }}
                    filePaths={pendingFiles}
                    onConfirm={handleUploadConfirm}
                />
            </Box>
        </Box >
    );
}
