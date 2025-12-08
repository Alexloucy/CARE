import React, { useState } from 'react';
import { Box, Typography, Collapse, IconButton } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { CaretDown, CaretRight, Code, CheckCircle, XCircle, Terminal } from '@phosphor-icons/react';
import { ToolResult } from '../../types/agent';

interface CodeExecutionBlockProps {
    result: ToolResult;
    toolName?: string;
}

const CodeExecutionBlock: React.FC<CodeExecutionBlockProps> = ({ result, toolName = 'Tool' }) => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    const [expanded, setExpanded] = useState(!result.success); // Auto-expand on error

    const isCodeExecution = toolName === 'runPythonCode';
    const displayName = isCodeExecution ? 'Python Code Execution' : toolName;
    const Icon = isCodeExecution ? Code : Terminal;

    const hasImages = result.images && result.images.length > 0;
    const hasCollapsibleContent = result.code || result.output || result.error;

    return (
        <Box
            sx={{
                borderRadius: 2,
                overflow: 'hidden',
                border: `1px solid ${result.success
                    ? (isDark ? 'rgba(76, 175, 80, 0.3)' : 'rgba(76, 175, 80, 0.4)')
                    : (isDark ? 'rgba(244, 67, 54, 0.3)' : 'rgba(244, 67, 54, 0.4)')}`,
                background: isDark
                    ? 'rgba(30, 30, 36, 0.5)'
                    : 'rgba(255, 255, 255, 0.5)',
                backdropFilter: 'blur(40px)',
                WebkitBackdropFilter: 'blur(40px)',
            }}
        >
            {/* Header - clickable to expand/collapse */}
            <Box
                onClick={() => hasCollapsibleContent && setExpanded(!expanded)}
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    p: 1.5,
                    cursor: hasCollapsibleContent ? 'pointer' : 'default',
                    background: result.success
                        ? (isDark ? 'rgba(76, 175, 80, 0.1)' : 'rgba(76, 175, 80, 0.08)')
                        : (isDark ? 'rgba(244, 67, 54, 0.1)' : 'rgba(244, 67, 54, 0.08)'),
                    '&:hover': hasCollapsibleContent ? {
                        background: result.success
                            ? (isDark ? 'rgba(76, 175, 80, 0.15)' : 'rgba(76, 175, 80, 0.12)')
                            : (isDark ? 'rgba(244, 67, 54, 0.15)' : 'rgba(244, 67, 54, 0.12)'),
                    } : {},
                }}
            >
                {hasCollapsibleContent && (
                    <IconButton size="small" sx={{ p: 0.25 }}>
                        {expanded ? <CaretDown size={16} /> : <CaretRight size={16} />}
                    </IconButton>
                )}
                <Icon size={18} weight="duotone" />
                <Typography variant="body2" fontWeight={500} sx={{ flex: 1 }}>
                    {displayName}
                </Typography>
                {result.success ? (
                    <CheckCircle size={18} weight="fill" color={theme.palette.success.main} />
                ) : (
                    <XCircle size={18} weight="fill" color={theme.palette.error.main} />
                )}
            </Box>

            {/* Collapsible content (code, output, error) */}
            {hasCollapsibleContent && (
                <Collapse in={expanded}>
                    <Box sx={{ p: 2 }}>
                        {/* Code section */}
                        {result.code && (
                            <>
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
                            </>
                        )}

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
                                    }}
                                >
                                    {result.output}
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
                                    }}
                                >
                                    {result.error}
                                </Box>
                            </>
                        )}
                    </Box>
                </Collapse>
            )}

            {/* Images section - ALWAYS shown outside collapse */}
            {hasImages && (
                <Box sx={{ p: 2, pt: expanded ? 0 : 2 }}>
                    <Typography variant="caption" color="text.secondary" fontWeight={600} sx={{ display: 'block', mb: 1 }}>
                        GENERATED CHARTS
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        {result.images!.map((imgSrc, idx) => (
                            <Box
                                key={idx}
                                component="img"
                                src={imgSrc}
                                alt={`Generated chart ${idx + 1}`}
                                sx={{
                                    maxWidth: '100%',
                                    borderRadius: 2,
                                    border: `1px solid ${isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'}`,
                                    backgroundColor: 'white',
                                }}
                            />
                        ))}
                    </Box>
                </Box>
            )}
        </Box>
    );
};

export default CodeExecutionBlock;
