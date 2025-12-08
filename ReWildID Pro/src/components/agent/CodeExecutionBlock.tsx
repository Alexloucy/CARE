import React, { useState } from 'react';
import { Box, Typography, Collapse, IconButton } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { CaretDown, CaretRight, Code, CheckCircle, XCircle } from '@phosphor-icons/react';

export interface CodeExecutionResult {
    success: boolean;
    error: string | null;
    output: string | null;
    images: string[];
    code: string;
}

interface CodeExecutionBlockProps {
    result: CodeExecutionResult;
}

const CodeExecutionBlock: React.FC<CodeExecutionBlockProps> = ({ result }) => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    const [expanded, setExpanded] = useState(!result.success); // Auto-expand on error

    return (
        <Box
            sx={{
                mt: 2,
                borderRadius: 2,
                overflow: 'hidden',
                border: `1px solid ${result.success
                    ? (isDark ? 'rgba(76, 175, 80, 0.3)' : 'rgba(76, 175, 80, 0.4)')
                    : (isDark ? 'rgba(244, 67, 54, 0.3)' : 'rgba(244, 67, 54, 0.4)')}`,
                background: isDark ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.03)',
            }}
        >
            {/* Header - clickable to expand/collapse */}
            <Box
                onClick={() => setExpanded(!expanded)}
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    p: 1.5,
                    cursor: 'pointer',
                    background: result.success
                        ? (isDark ? 'rgba(76, 175, 80, 0.1)' : 'rgba(76, 175, 80, 0.08)')
                        : (isDark ? 'rgba(244, 67, 54, 0.1)' : 'rgba(244, 67, 54, 0.08)'),
                    '&:hover': {
                        background: result.success
                            ? (isDark ? 'rgba(76, 175, 80, 0.15)' : 'rgba(76, 175, 80, 0.12)')
                            : (isDark ? 'rgba(244, 67, 54, 0.15)' : 'rgba(244, 67, 54, 0.12)'),
                    },
                }}
            >
                <IconButton size="small" sx={{ p: 0.25 }}>
                    {expanded ? <CaretDown size={16} /> : <CaretRight size={16} />}
                </IconButton>
                <Code size={18} weight="duotone" />
                <Typography variant="body2" fontWeight={500} sx={{ flex: 1 }}>
                    Python Code Execution
                </Typography>
                {result.success ? (
                    <CheckCircle size={18} weight="fill" color={theme.palette.success.main} />
                ) : (
                    <XCircle size={18} weight="fill" color={theme.palette.error.main} />
                )}
            </Box>

            {/* Expandable content */}
            <Collapse in={expanded}>
                <Box sx={{ p: 2 }}>
                    {/* Code section */}
                    <Typography variant="caption" color="text.secondary" fontWeight={600} sx={{ display: 'block', mb: 0.5 }}>
                        CODE
                    </Typography>
                    <Box
                        sx={{
                            p: 1.5,
                            borderRadius: 1,
                            fontFamily: 'monospace',
                            fontSize: '0.8rem',
                            whiteSpace: 'pre-wrap',
                            overflowX: 'auto',
                            background: isDark ? 'rgba(0,0,0,0.4)' : 'rgba(0,0,0,0.06)',
                            color: isDark ? '#e0e0e0' : '#333',
                            maxHeight: 200,
                            overflow: 'auto',
                            mb: 2,
                        }}
                    >
                        {result.code}
                    </Box>

                    {/* Output section */}
                    {result.output && (
                        <>
                            <Typography variant="caption" color="text.secondary" fontWeight={600} sx={{ display: 'block', mb: 0.5 }}>
                                OUTPUT
                            </Typography>
                            <Box
                                sx={{
                                    p: 1.5,
                                    borderRadius: 1,
                                    fontFamily: 'monospace',
                                    fontSize: '0.75rem',
                                    whiteSpace: 'pre-wrap',
                                    background: isDark ? 'rgba(0,0,0,0.4)' : 'rgba(0,0,0,0.06)',
                                    color: isDark ? '#b0b0b0' : '#555',
                                    maxHeight: 150,
                                    overflow: 'auto',
                                    mb: result.images.length > 0 ? 2 : 0,
                                }}
                            >
                                {result.output || '(no output)'}
                            </Box>
                        </>
                    )}

                    {/* Error section */}
                    {result.error && (
                        <>
                            <Typography variant="caption" color="error" fontWeight={600} sx={{ display: 'block', mb: 0.5 }}>
                                ERROR
                            </Typography>
                            <Box
                                sx={{
                                    p: 1.5,
                                    borderRadius: 1,
                                    fontFamily: 'monospace',
                                    fontSize: '0.75rem',
                                    whiteSpace: 'pre-wrap',
                                    background: isDark ? 'rgba(244,67,54,0.1)' : 'rgba(244,67,54,0.08)',
                                    color: theme.palette.error.main,
                                    mb: result.images.length > 0 ? 2 : 0,
                                }}
                            >
                                {result.error}
                            </Box>
                        </>
                    )}

                    {/* Images section */}
                    {result.images.length > 0 && (
                        <>
                            <Typography variant="caption" color="text.secondary" fontWeight={600} sx={{ display: 'block', mb: 1 }}>
                                GENERATED CHARTS
                            </Typography>
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                {result.images.map((imgSrc, idx) => (
                                    <Box
                                        key={idx}
                                        component="img"
                                        src={imgSrc}
                                        alt={`Generated chart ${idx + 1}`}
                                        sx={{
                                            maxWidth: '100%',
                                            borderRadius: 2,
                                            border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
                                            backgroundColor: 'white', // Charts usually look better on white
                                        }}
                                    />
                                ))}
                            </Box>
                        </>
                    )}
                </Box>
            </Collapse>
        </Box>
    );
};

export default CodeExecutionBlock;
