import React, { useState } from 'react';
import {
    Box,
    Dialog,
    DialogTitle,
    DialogContent,
    Typography,
    IconButton,
    alpha,
    useTheme,
    MenuItem,
    Select,
    FormControl,
    InputLabel,
    Button,
    Divider
} from '@mui/material';
import { X, Sparkle, Fingerprint } from '@phosphor-icons/react';

interface AnalyseMenuProps {
    open: boolean;
    onClose: () => void;
    onClassify: () => void;
    onReID: (species: string) => void;
    availableSpecies: string[];
    selectedCount: number;
}

export const AnalyseMenu: React.FC<AnalyseMenuProps> = ({
    open,
    onClose,
    onClassify,
    onReID,
    availableSpecies,
    selectedCount
}) => {
    const theme = useTheme();
    const [selectedSpecies, setSelectedSpecies] = useState<string>('');
    const [showReIDOptions, setShowReIDOptions] = useState(false);

    const handleClassify = () => {
        onClassify();
        onClose();
    };

    const handleReID = () => {
        if (selectedSpecies) {
            onReID(selectedSpecies);
            onClose();
            setShowReIDOptions(false);
            setSelectedSpecies('');
        }
    };

    const handleClose = () => {
        onClose();
        setShowReIDOptions(false);
        setSelectedSpecies('');
    };

    return (
        <Dialog
            open={open}
            onClose={handleClose}
            maxWidth="xs"
            fullWidth
            PaperProps={{
                sx: {
                    bgcolor: theme.palette.mode === 'light' 
                        ? alpha('#FFFFFF', 0.85) 
                        : alpha(theme.palette.background.paper, 0.85),
                    backdropFilter: 'blur(20px)',
                    borderRadius: 3,
                    border: `1px solid ${alpha(theme.palette.divider, 0.3)}`,
                    boxShadow: theme.palette.mode === 'light'
                        ? '0 8px 32px rgba(0, 0, 0, 0.12)'
                        : '0 8px 32px rgba(0, 0, 0, 0.4)',
                    overflow: 'hidden'
                }
            }}
        >
            <DialogTitle sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between',
                pb: 1
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <Sparkle size={24} weight="duotone" />
                    <Typography variant="h6" fontWeight={600}>
                        Analyse
                    </Typography>
                </Box>
                <IconButton onClick={handleClose} size="small">
                    <X />
                </IconButton>
            </DialogTitle>

            <DialogContent sx={{ pt: 1 }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2.5 }}>
                    {selectedCount} image{selectedCount !== 1 ? 's' : ''} selected
                </Typography>

                {!showReIDOptions ? (
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                        {/* Classification Option */}
                        <Box
                            onClick={handleClassify}
                            sx={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 2,
                                p: 2,
                                borderRadius: 2,
                                cursor: 'pointer',
                                bgcolor: alpha(theme.palette.primary.main, 0.08),
                                border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                                transition: 'all 0.2s ease',
                                '&:hover': {
                                    bgcolor: alpha(theme.palette.primary.main, 0.15),
                                    transform: 'translateY(-1px)',
                                    boxShadow: `0 4px 12px ${alpha(theme.palette.primary.main, 0.2)}`
                                }
                            }}
                        >
                            <Box sx={{
                                width: 44,
                                height: 44,
                                borderRadius: 2,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                bgcolor: alpha(theme.palette.primary.main, 0.15),
                                color: theme.palette.primary.main
                            }}>
                                <Sparkle size={24} weight="fill" />
                            </Box>
                            <Box>
                                <Typography fontWeight={600}>Classification</Typography>
                                <Typography variant="caption" color="text.secondary">
                                    Detect and classify animals in images
                                </Typography>
                            </Box>
                        </Box>

                        {/* Re-identification Option */}
                        <Box
                            onClick={() => setShowReIDOptions(true)}
                            sx={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 2,
                                p: 2,
                                borderRadius: 2,
                                cursor: 'pointer',
                                bgcolor: alpha(theme.palette.secondary.main, 0.08),
                                border: `1px solid ${alpha(theme.palette.secondary.main, 0.2)}`,
                                transition: 'all 0.2s ease',
                                '&:hover': {
                                    bgcolor: alpha(theme.palette.secondary.main, 0.15),
                                    transform: 'translateY(-1px)',
                                    boxShadow: `0 4px 12px ${alpha(theme.palette.secondary.main, 0.2)}`
                                }
                            }}
                        >
                            <Box sx={{
                                width: 44,
                                height: 44,
                                borderRadius: 2,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                bgcolor: alpha(theme.palette.secondary.main, 0.15),
                                color: theme.palette.secondary.main
                            }}>
                                <Fingerprint size={24} weight="fill" />
                            </Box>
                            <Box>
                                <Typography fontWeight={600}>Re-identification</Typography>
                                <Typography variant="caption" color="text.secondary">
                                    Match individuals across images
                                </Typography>
                            </Box>
                        </Box>
                    </Box>
                ) : (
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1 }}>
                            <Fingerprint size={20} weight="fill" color={theme.palette.secondary.main} />
                            <Typography fontWeight={600}>Re-identification</Typography>
                        </Box>
                        
                        <Typography variant="body2" color="text.secondary">
                            Select a species to identify individuals
                        </Typography>

                        <FormControl fullWidth size="small">
                            <InputLabel>Species</InputLabel>
                            <Select
                                value={selectedSpecies}
                                onChange={(e) => setSelectedSpecies(e.target.value)}
                                label="Species"
                                sx={{ borderRadius: 2 }}
                            >
                                {availableSpecies.length === 0 ? (
                                    <MenuItem disabled>
                                        <em>No species detected yet</em>
                                    </MenuItem>
                                ) : (
                                    availableSpecies.map(species => (
                                        <MenuItem key={species} value={species}>
                                            {species}
                                        </MenuItem>
                                    ))
                                )}
                            </Select>
                        </FormControl>

                        <Divider sx={{ my: 1 }} />

                        <Box sx={{ display: 'flex', gap: 1.5, justifyContent: 'flex-end' }}>
                            <Button 
                                variant="text" 
                                onClick={() => setShowReIDOptions(false)}
                                sx={{ borderRadius: 2 }}
                            >
                                Back
                            </Button>
                            <Button
                                variant="contained"
                                onClick={handleReID}
                                disabled={!selectedSpecies}
                                startIcon={<Fingerprint size={18} />}
                                sx={{ 
                                    borderRadius: 2,
                                    textTransform: 'none',
                                    bgcolor: theme.palette.secondary.main,
                                    '&:hover': {
                                        bgcolor: theme.palette.secondary.dark
                                    }
                                }}
                            >
                                Start ReID
                            </Button>
                        </Box>
                    </Box>
                )}
            </DialogContent>
        </Dialog>
    );
};
