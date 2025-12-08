import React, { useState, useEffect } from 'react';
import {
    Box,
    Container,
    Typography,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Table,
    TableBody,
    TableRow,
    TableCell,
    TableContainer,
    Slider,
    ToggleButton,
    ToggleButtonGroup,
    IconButton,
    Tooltip,
    useTheme,
    alpha,
    TextField,
    InputAdornment,
} from '@mui/material';
import { getAgentSettings, saveAgentSettings } from '../../services/agentService';
import {
    CaretDown,
    GridFour,
    Eye,
    Sparkle,
    Tag,
    Fingerprint,
    ArrowCounterClockwise,
    Image as ImageIcon,
    Cube,
    TextT,
    Drop,
    BoundingBox,
    Lightbulb,
    PaintBrush,
    Check,
    Sun,
    Moon,
    Trash,
    Robot,
    Key,
} from '@phosphor-icons/react';
import StyledSwitch from '../../components/StyledSwitch';
import { resetAllTours } from '../../components/OnboardingTour';
import { useColorMode, COLOR_THEMES, ColorTheme } from '../../features/theme/ThemeContext';

// Settings item interface
interface SettingsItem {
    key: string;
    label: string;
    description?: string;
    icon: JSX.Element;
    section: 'display' | 'visual' | 'tags';
    type: 'switch' | 'slider' | 'toggle';
    options?: string[];
    min?: number;
    max?: number;
    defaultValue?: any;
}

// All settings items configuration
const settingsItems: SettingsItem[] = [
    // Display Settings
    {
        key: 'gridSize',
        label: 'Grid Size',
        description: 'Adjust the size of image thumbnails in the library grid',
        icon: <GridFour size={24} />,
        section: 'display',
        type: 'slider',
        min: 100,
        max: 715,
        defaultValue: 180
    },
    {
        key: 'aspectRatio',
        label: 'Aspect Ratio',
        description: 'Set the aspect ratio for image thumbnails',
        icon: <ImageIcon size={24} />,
        section: 'display',
        type: 'toggle',
        options: ['1/1', '4/3', '16/9', '9/16', '1.618/1', '1/1.618'],
        defaultValue: '1.618/1'
    },
    {
        key: 'showNames',
        label: 'Show File Names',
        description: 'Display file names below image thumbnails',
        icon: <TextT size={24} />,
        section: 'display',
        type: 'switch',
        defaultValue: false
    },
    // Visual Effects
    {
        key: 'useLiquidGlass',
        label: 'Liquid Glass BBox',
        description: 'Use liquid glass effect for detection bounding boxes',
        icon: <Drop size={24} />,
        section: 'visual',
        type: 'switch',
        defaultValue: true
    },
    {
        key: 'useRayTracedGlass',
        label: 'Ray-traced Glass',
        description: 'Enable ray-traced rendering for liquid glass effect (requires Liquid Glass to be enabled)',
        icon: <Cube size={24} />,
        section: 'visual',
        type: 'switch',
        defaultValue: true
    },
    // Tags
    {
        key: 'showSpeciesTags',
        label: 'Show Species Tags',
        description: 'Display species classification tags on image thumbnails',
        icon: <Tag size={24} />,
        section: 'tags',
        type: 'switch',
        defaultValue: true
    },
    {
        key: 'showReidTags',
        label: 'Show Individual Tags',
        description: 'Display re-identification individual name tags on image thumbnails',
        icon: <Fingerprint size={24} />,
        section: 'tags',
        type: 'switch',
        defaultValue: true
    },
    {
        key: 'showBoundingBoxes',
        label: 'Show Bounding Boxes',
        description: 'Display detection bounding boxes when viewing images in full screen',
        icon: <BoundingBox size={24} />,
        section: 'tags',
        type: 'switch',
        defaultValue: true
    },
];

// Settings sections configuration
const sections = [
    {
        id: 'display',
        title: 'Display Settings',
        description: 'Customize how images are displayed in the library',
        icon: <Eye size={20} />
    },
    {
        id: 'visual',
        title: 'Visual Effects',
        description: 'Configure visual effects for detection overlays',
        icon: <Sparkle size={20} />
    },
    {
        id: 'tags',
        title: 'Tag Visibility',
        description: 'Control which tags are shown on image thumbnails',
        icon: <Tag size={20} />
    },
];

// Theme swatch component
const ThemeSwatch: React.FC<{
    themeOption: ColorTheme;
    isSelected: boolean;
    onClick: () => void;
}> = ({ themeOption, isSelected, onClick }) => {
    const muiTheme = useTheme();

    return (
        <Tooltip title={`${themeOption.name} (${themeOption.mode === 'dark' ? 'Dark' : 'Light'})`}>
            <Box
                onClick={onClick}
                sx={{
                    width: 64,
                    height: 64,
                    borderRadius: 2,
                    background: themeOption.previewGradient,
                    cursor: 'pointer',
                    position: 'relative',
                    border: isSelected
                        ? `3px solid ${muiTheme.palette.primary.main}`
                        : `2px solid ${alpha(muiTheme.palette.divider, 0.3)}`,
                    transition: 'all 0.2s ease',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    overflow: 'hidden',
                    '&:hover': {
                        transform: 'scale(1.05)',
                        borderColor: muiTheme.palette.primary.main,
                    },
                }}
            >
                {/* Mode indicator */}
                <Box
                    sx={{
                        position: 'absolute',
                        bottom: 4,
                        right: 4,
                        width: 18,
                        height: 18,
                        borderRadius: '50%',
                        backgroundColor: themeOption.mode === 'dark'
                            ? 'rgba(0,0,0,0.7)'
                            : 'rgba(255,255,255,0.9)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        border: `1px solid ${themeOption.mode === 'dark' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.1)'}`,
                    }}
                >
                    {themeOption.mode === 'dark' ? (
                        <Moon size={10} weight="fill" color="#fff" />
                    ) : (
                        <Sun size={10} weight="fill" color="#000" />
                    )}
                </Box>

                {/* Selected checkmark */}
                {isSelected && (
                    <Box
                        sx={{
                            position: 'absolute',
                            top: 4,
                            left: 4,
                            width: 20,
                            height: 20,
                            borderRadius: '50%',
                            backgroundColor: muiTheme.palette.primary.main,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                        }}
                    >
                        <Check size={12} weight="bold" color="#fff" />
                    </Box>
                )}
            </Box>
        </Tooltip>
    );
};

const SettingsPage: React.FC = () => {
    const theme = useTheme();
    const { colorTheme, setColorTheme } = useColorMode();
    const hasGradient = colorTheme.gradient !== 'none' || !!colorTheme.special || !!colorTheme.image;

    // AI Agent settings
    const [aiApiKey, setAiApiKey] = useState(() => getAgentSettings().apiKey);
    const [showApiKey, setShowApiKey] = useState(false);

    const handleApiKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newKey = e.target.value;
        setAiApiKey(newKey);
        saveAgentSettings({ apiKey: newKey });
    };

    // Load all settings from localStorage
    const [settings, setSettings] = useState<Record<string, any>>(() => {
        const initial: Record<string, any> = {};
        settingsItems.forEach(item => {
            const storageKey = `mediaExplorer_${item.key}`;
            const saved = localStorage.getItem(storageKey);
            if (saved !== null) {
                if (item.type === 'switch') {
                    initial[item.key] = saved === 'true';
                } else if (item.type === 'slider') {
                    initial[item.key] = parseInt(saved, 10);
                } else {
                    initial[item.key] = saved;
                }
            } else {
                initial[item.key] = item.defaultValue;
            }
        });
        return initial;
    });

    // Persist settings to localStorage
    useEffect(() => {
        Object.entries(settings).forEach(([key, value]) => {
            localStorage.setItem(`mediaExplorer_${key}`, value.toString());
        });
    }, [settings]);

    // Update a single setting
    const updateSetting = (key: string, value: any) => {
        setSettings(prev => ({ ...prev, [key]: value }));
    };

    // Reset a single setting to default
    const resetSetting = (key: string) => {
        const item = settingsItems.find(i => i.key === key);
        if (item) {
            updateSetting(key, item.defaultValue);
        }
    };

    // Render a setting row based on its type
    const renderSettingRow = (item: SettingsItem) => {
        const value = settings[item.key];
        const isDisabled = item.key === 'useRayTracedGlass' && !settings['useLiquidGlass'];

        return (
            <TableRow key={item.key} sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
                <TableCell sx={{ width: 50, pr: 1, borderBottom: 'none', opacity: isDisabled ? 0.5 : 1 }}>
                    {React.cloneElement(item.icon, { color: theme.palette.text.secondary })}
                </TableCell>
                <TableCell sx={{ borderBottom: 'none', opacity: isDisabled ? 0.5 : 1 }}>
                    <Typography variant="body2" fontWeight="medium">
                        {item.label}
                    </Typography>
                    {item.description && (
                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.25 }}>
                            {item.description}
                        </Typography>
                    )}
                </TableCell>
                <TableCell align="right" sx={{ borderBottom: 'none', minWidth: item.type === 'toggle' ? 280 : 120 }}>
                    {item.type === 'switch' && (
                        <StyledSwitch
                            checked={Boolean(value)}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateSetting(item.key, e.target.checked)}
                            disabled={isDisabled}
                            inputProps={{ 'aria-label': `${item.label} toggle` }}
                        />
                    )}
                    {item.type === 'slider' && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Slider
                                size="small"
                                value={value}
                                min={item.min}
                                max={item.max}
                                onChange={(_: Event, newValue: number | number[]) => updateSetting(item.key, newValue as number)}
                                valueLabelDisplay="auto"
                                valueLabelFormat={(v: number) => `${v}px`}
                                sx={{ width: 150 }}
                            />
                            <Tooltip title="Reset to Default">
                                <IconButton size="small" onClick={() => resetSetting(item.key)}>
                                    <ArrowCounterClockwise size={14} />
                                </IconButton>
                            </Tooltip>
                        </Box>
                    )}
                    {item.type === 'toggle' && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <ToggleButtonGroup
                                value={value}
                                exclusive
                                onChange={(_: React.MouseEvent<HTMLElement>, newValue: string | null) => newValue && updateSetting(item.key, newValue)}
                                size="small"
                            >
                                {item.options?.map(opt => (
                                    <ToggleButton
                                        key={opt}
                                        value={opt}
                                        sx={{
                                            py: 0.5,
                                            px: 1.5,
                                            fontSize: '0.75rem'
                                        }}
                                    >
                                        {opt === '1.618/1' || opt === '1/1.618' ? 'Φ' : opt.replace('/', ':')}
                                    </ToggleButton>
                                ))}
                            </ToggleButtonGroup>
                            <Tooltip title="Reset to Default">
                                <IconButton size="small" onClick={() => resetSetting(item.key)}>
                                    <ArrowCounterClockwise size={14} />
                                </IconButton>
                            </Tooltip>
                        </Box>
                    )}
                </TableCell>
            </TableRow>
        );
    };

    return (
        <Box sx={{ pt: '80px', minHeight: '100vh' }}>
            <Container maxWidth="md" sx={{ pb: 4 }}>
                {/* Page Title */}
                <Typography variant="h4" fontWeight="bold" sx={{ mb: 3 }}>
                    Settings
                </Typography>

                {/* Theme Section */}
                <Accordion
                    defaultExpanded
                    disableGutters
                    square
                    elevation={0}
                    sx={{
                        mb: 2,
                        border: `1px solid ${theme.palette.divider}`,
                        borderRadius: 2.5,
                        '&:before': { display: 'none' },
                        overflow: 'hidden',
                        bgcolor: hasGradient
                            ? (theme.palette.mode === 'dark' ? 'rgba(30, 30, 36, 0.75)' : 'rgba(247, 249, 251, 0.75)')
                            : 'background.paper',
                        backdropFilter: hasGradient ? 'blur(12px)' : 'none',
                        WebkitBackdropFilter: hasGradient ? 'blur(12px)' : 'none',
                    }}
                >
                    <AccordionSummary
                        expandIcon={<CaretDown size={20} />}
                        sx={{
                            px: 2,
                            minHeight: '56px',
                            '& .MuiAccordionSummary-content': {
                                my: '12px'
                            },
                            '&.Mui-expanded': {
                                minHeight: '56px',
                                borderBottom: `1px solid ${theme.palette.divider}`
                            }
                        }}
                    >
                        <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%', pr: 1 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <PaintBrush size={20} />
                                <Typography variant="h6" fontWeight="600">
                                    Theme
                                </Typography>
                            </Box>
                            <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{ mt: 0, lineHeight: 1.3, ml: 3.5 }}
                            >
                                Choose a color theme for the application
                            </Typography>
                        </Box>
                    </AccordionSummary>
                    <AccordionDetails sx={{ pt: 2, px: 2, pb: 2 }}>
                        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                            {COLOR_THEMES.map((themeOption) => (
                                <Box key={themeOption.id} sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.5 }}>
                                    <ThemeSwatch
                                        themeOption={themeOption}
                                        isSelected={colorTheme.id === themeOption.id}
                                        onClick={() => setColorTheme(themeOption.id)}
                                    />
                                    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                                        {themeOption.name}
                                    </Typography>
                                </Box>
                            ))}
                        </Box>
                    </AccordionDetails>
                </Accordion>

                {/* AI Agent Settings */}
                <Accordion
                    defaultExpanded
                    disableGutters
                    square
                    elevation={0}
                    sx={{
                        mb: 2,
                        border: `1px solid ${theme.palette.divider}`,
                        borderRadius: 2.5,
                        '&:before': { display: 'none' },
                        overflow: 'hidden',
                        bgcolor: hasGradient
                            ? (theme.palette.mode === 'dark' ? 'rgba(30, 30, 36, 0.75)' : 'rgba(247, 249, 251, 0.75)')
                            : 'background.paper',
                        backdropFilter: hasGradient ? 'blur(12px)' : 'none',
                        WebkitBackdropFilter: hasGradient ? 'blur(12px)' : 'none',
                    }}
                >
                    <AccordionSummary
                        expandIcon={<CaretDown size={20} />}
                        sx={{
                            px: 2,
                            minHeight: '56px',
                            '& .MuiAccordionSummary-content': {
                                my: '12px'
                            },
                            '&.Mui-expanded': {
                                minHeight: '56px',
                                borderBottom: `1px solid ${theme.palette.divider}`
                            }
                        }}
                    >
                        <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%', pr: 1 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Robot size={20} />
                                <Typography variant="h6" fontWeight="600">
                                    AI Agent
                                </Typography>
                            </Box>
                            <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{ mt: 0, lineHeight: 1.3, ml: 3.5 }}
                            >
                                Configure your Google AI Studio API key for the AI agent
                            </Typography>
                        </Box>
                    </AccordionSummary>
                    <AccordionDetails sx={{ pt: 2, px: 2, pb: 2 }}>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                                <Key size={24} style={{ marginTop: 8 }} color={theme.palette.text.secondary} />
                                <Box sx={{ flex: 1 }}>
                                    <Typography variant="body2" fontWeight="medium" sx={{ mb: 0.5 }}>
                                        Google AI Studio API Key
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                                        Get your API key from{' '}
                                        <a
                                            href="https://aistudio.google.com/apikey"
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            style={{ color: theme.palette.primary.main }}
                                        >
                                            aistudio.google.com
                                        </a>
                                    </Typography>
                                    <TextField
                                        fullWidth
                                        size="small"
                                        type={showApiKey ? 'text' : 'password'}
                                        value={aiApiKey}
                                        onChange={handleApiKeyChange}
                                        placeholder="AIza..."
                                        InputProps={{
                                            endAdornment: (
                                                <InputAdornment position="end">
                                                    <IconButton
                                                        size="small"
                                                        onClick={() => setShowApiKey(!showApiKey)}
                                                        edge="end"
                                                    >
                                                        <Eye size={18} weight={showApiKey ? 'fill' : 'regular'} />
                                                    </IconButton>
                                                </InputAdornment>
                                            ),
                                        }}
                                        sx={{
                                            '& .MuiOutlinedInput-root': {
                                                borderRadius: 2,
                                            }
                                        }}
                                    />
                                </Box>
                            </Box>
                            <Typography variant="caption" color="text.secondary">
                                Using model: <strong>gemini-flash-latest</strong>
                            </Typography>
                        </Box>
                    </AccordionDetails>
                </Accordion>

                {/* Settings Sections */}
                {sections.map(({ id, title, description }) => (
                    <Accordion
                        key={id}
                        defaultExpanded
                        disableGutters
                        square
                        elevation={0}
                        sx={{
                            mb: 2,
                            border: `1px solid ${theme.palette.divider}`,
                            borderRadius: 2.5,
                            '&:before': { display: 'none' },
                            overflow: 'hidden',
                            bgcolor: hasGradient
                                ? (theme.palette.mode === 'dark' ? 'rgba(30, 30, 36, 0.75)' : 'rgba(247, 249, 251, 0.75)')
                                : 'background.paper',
                            backdropFilter: hasGradient ? 'blur(12px)' : 'none',
                            WebkitBackdropFilter: hasGradient ? 'blur(12px)' : 'none',
                        }}
                    >
                        <AccordionSummary
                            expandIcon={<CaretDown size={20} />}
                            sx={{
                                px: 2,
                                minHeight: '56px',
                                '& .MuiAccordionSummary-content': {
                                    my: '12px'
                                },
                                '&.Mui-expanded': {
                                    minHeight: '56px',
                                    borderBottom: `1px solid ${theme.palette.divider}`
                                }
                            }}
                        >
                            <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%', pr: 1 }}>
                                <Typography variant="h6" fontWeight="600">
                                    {title}
                                </Typography>
                                {description && (
                                    <Typography
                                        variant="caption"
                                        color="text.secondary"
                                        sx={{ mt: 0, lineHeight: 1.3 }}
                                    >
                                        {description}
                                    </Typography>
                                )}
                            </Box>
                        </AccordionSummary>
                        <AccordionDetails sx={{ pt: 1, px: 2, pb: 2 }}>
                            <TableContainer>
                                <Table>
                                    <TableBody>
                                        {settingsItems
                                            .filter(item => item.section === id)
                                            .map(renderSettingRow)}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </AccordionDetails>
                    </Accordion>
                ))}

                {/* Reset Onboarding Tours */}
                <Box
                    sx={{
                        mt: 3,
                        p: 2,
                        border: `1px solid ${theme.palette.divider}`,
                        borderRadius: 2.5,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        bgcolor: hasGradient
                            ? (theme.palette.mode === 'dark' ? 'rgba(30, 30, 36, 0.75)' : 'rgba(247, 249, 251, 0.75)')
                            : 'transparent',
                        backdropFilter: hasGradient ? 'blur(12px)' : 'none',
                        WebkitBackdropFilter: hasGradient ? 'blur(12px)' : 'none',
                    }}
                >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Lightbulb size={24} />
                        <Box>
                            <Typography variant="body1" fontWeight={500}>Reset Guided Tours</Typography>
                            <Typography variant="caption" color="text.secondary">
                                Show the onboarding walkthrough again for all pages
                            </Typography>
                        </Box>
                    </Box>
                    <Tooltip title="Reset Tours">
                        <IconButton
                            onClick={() => {
                                resetAllTours();
                                alert('Tours have been reset! You will see them again when visiting each page.');
                            }}
                            sx={{
                                bgcolor: theme.palette.action.hover,
                                '&:hover': { bgcolor: theme.palette.action.selected }
                            }}
                        >
                            <ArrowCounterClockwise size={20} />
                        </IconButton>
                    </Tooltip>
                </Box>

                {/* Clear Embeddings Cache */}
                <Box
                    sx={{
                        mt: 2,
                        p: 2,
                        border: `1px solid ${theme.palette.divider}`,
                        borderRadius: 2.5,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        bgcolor: hasGradient
                            ? (theme.palette.mode === 'dark' ? 'rgba(30, 30, 36, 0.75)' : 'rgba(247, 249, 251, 0.75)')
                            : 'transparent',
                        backdropFilter: hasGradient ? 'blur(12px)' : 'none',
                        WebkitBackdropFilter: hasGradient ? 'blur(12px)' : 'none',
                    }}
                >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Cube size={24} />
                        <Box>
                            <Typography variant="body1" fontWeight={500}>Clear Embeddings Cache</Typography>
                            <Typography variant="caption" color="text.secondary">
                                Remove cached AI embeddings to free disk space. Classification and Re-ID will take longer after clearing.
                            </Typography>
                        </Box>
                    </Box>
                    <Tooltip title="Clear Cache">
                        <IconButton
                            onClick={async () => {
                                if (confirm('Are you sure you want to clear all cached embeddings? This cannot be undone and Classification/Re-ID jobs will take longer.')) {
                                    const result = await window.api.clearEmbeddingsCache();
                                    if (result.ok) {
                                        alert(`Cleared ${result.count} cached embeddings.`);
                                    } else {
                                        alert('Failed to clear cache: ' + result.error);
                                    }
                                }
                            }}
                            sx={{
                                bgcolor: theme.palette.action.hover,
                                '&:hover': { bgcolor: theme.palette.error.main, color: 'white' }
                            }}
                        >
                            <Trash size={20} />
                        </IconButton>
                    </Tooltip>
                </Box>
            </Container>
        </Box>
    );
};

export default SettingsPage;
